"""
CIFT Markets - Price Alerts & Monitoring Service

Real-time price monitoring with intelligent alert system.
Supports multiple alert types, conditions, and notification methods.
"""

import asyncio
import json
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID, uuid4

from loguru import logger
from pydantic import BaseModel

from cift.core.database import get_postgres_pool


class AlertType(str, Enum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PRICE_CHANGE_PERCENT = "price_change_percent"
    VOLUME_SPIKE = "volume_spike"
    TECHNICAL_INDICATOR = "technical_indicator"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXPIRED = "expired"
    DISABLED = "disabled"


class NotificationMethod(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"


class PriceAlert(BaseModel):
    id: UUID | None = None
    user_id: UUID
    symbol: str
    alert_type: AlertType
    condition_value: float
    condition_value2: float | None = None  # For range alerts
    notification_methods: list[NotificationMethod]
    message: str | None = None
    is_active: bool = True
    expires_at: datetime | None = None
    created_at: datetime | None = None
    triggered_at: datetime | None = None
    trigger_price: float | None = None


class AlertTrigger(BaseModel):
    alert_id: UUID
    symbol: str
    trigger_price: float
    trigger_time: datetime
    condition_met: str
    user_id: UUID
    notification_methods: list[str]
    message: str


class PriceAlertService:
    """Real-time price alert monitoring and notification service."""

    def __init__(self):
        self.monitoring = False
        self.check_interval = 5  # Check every 5 seconds
        self.active_alerts: dict[str, list[PriceAlert]] = {}  # symbol -> alerts

    async def start_monitoring(self):
        """Start the price monitoring service."""
        if self.monitoring:
            return

        self.monitoring = True
        logger.info("🚨 Starting price alert monitoring service...")

        # Load active alerts
        await self._load_active_alerts()

        # Start monitoring loop
        while self.monitoring:
            try:
                await self._check_alerts()
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                await asyncio.sleep(30)  # Wait longer on error

    def stop_monitoring(self):
        """Stop the price monitoring service."""
        self.monitoring = False
        logger.info("⏹️ Stopping price alert monitoring service...")

    async def create_alert(self, alert: PriceAlert) -> UUID:
        """Create a new price alert."""

        if not alert.id:
            alert.id = uuid4()

        if not alert.created_at:
            alert.created_at = datetime.utcnow()

        # Set default expiration (30 days)
        if not alert.expires_at:
            alert.expires_at = alert.created_at + timedelta(days=30)

        # Validate alert
        await self._validate_alert(alert)

        # Store in database
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO price_alerts (
                    id, user_id, symbol, alert_type, target_value,
                    notification_methods, status, expires_at, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                alert.id,
                alert.user_id,
                alert.symbol.upper(),
                alert.alert_type.value,
                alert.condition_value,
                alert.notification_methods,  # Pass list directly for text[]
                "active" if alert.is_active else "disabled",
                alert.expires_at,
                alert.created_at,
            )

        # Add to active monitoring
        if alert.is_active:
            await self._add_to_monitoring(alert)

        logger.info(f"Created price alert {alert.id} for {alert.symbol}")
        return alert.id

    async def update_alert(self, alert_id: UUID, updates: dict) -> bool:
        """Update an existing alert."""

        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            # Get current alert
            row = await conn.fetchrow(
                """
                SELECT * FROM price_alerts WHERE id = $1
            """,
                alert_id,
            )

            if not row:
                return False

            # Build update query
            update_fields = []
            params = []
            param_count = 1

            for field, value in updates.items():
                if field == "condition_value":
                    update_fields.append(f"target_value = ${param_count}")
                    params.append(value)
                    param_count += 1
                elif field == "is_active":
                    update_fields.append(f"status = ${param_count}")
                    params.append("active" if value else "disabled")
                    param_count += 1
                elif field == "notification_methods":
                    update_fields.append(f"notification_methods = ${param_count}")
                    # Ensure it's a list of strings
                    if isinstance(value, list):
                        params.append(value)
                    else:
                        params.append([])
                    param_count += 1
                elif field in ["alert_type", "expires_at"]:
                    update_fields.append(f"{field} = ${param_count}")
                    params.append(value)
                    param_count += 1

            if not update_fields:
                return False

            params.append(alert_id)

            await conn.execute(
                f"""
                UPDATE price_alerts
                SET {', '.join(update_fields)}
                WHERE id = ${param_count}
            """,
                *params,
            )

        # Reload alerts for this symbol
        symbol = row["symbol"]
        await self._reload_symbol_alerts(symbol)

        logger.info(f"Updated price alert {alert_id}")
        return True

    async def delete_alert(self, alert_id: UUID, user_id: UUID) -> bool:
        """Delete a price alert."""

        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            # Get alert info before deletion
            row = await conn.fetchrow(
                """
                SELECT symbol FROM price_alerts
                WHERE id = $1 AND user_id = $2
            """,
                alert_id,
                user_id,
            )

            if not row:
                return False

            # Delete the alert
            result = await conn.execute(
                """
                DELETE FROM price_alerts
                WHERE id = $1 AND user_id = $2
            """,
                alert_id,
                user_id,
            )

            if result == "DELETE 0":
                return False

        # Remove from active monitoring
        symbol = row["symbol"]
        if symbol in self.active_alerts:
            self.active_alerts[symbol] = [
                alert for alert in self.active_alerts[symbol] if alert.id != alert_id
            ]

            if not self.active_alerts[symbol]:
                del self.active_alerts[symbol]

        logger.info(f"Deleted price alert {alert_id}")
        return True

    async def get_user_alerts(
        self, user_id: UUID, include_triggered: bool = True
    ) -> list[PriceAlert]:
        """Get all alerts for a user."""

        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            query = """
                SELECT * FROM price_alerts
                WHERE user_id = $1
            """

            if not include_triggered:
                query += " AND triggered_at IS NULL"

            query += " ORDER BY created_at DESC"

            rows = await conn.fetch(query, user_id)

        alerts = []
        for row in rows:
            # Handle notification_methods (Postgres array)
            methods_list = row["notification_methods"] or []

            notification_methods = [
                NotificationMethod(method)
                for method in methods_list
                if method in [m.value for m in NotificationMethod]
            ]

            alerts.append(
                PriceAlert(
                    id=row["id"],
                    user_id=row["user_id"],
                    symbol=row["symbol"],
                    alert_type=AlertType(row["alert_type"]),
                    condition_value=float(row["target_value"]),
                    condition_value2=None,
                    notification_methods=notification_methods,
                    message=None,
                    is_active=row["status"] == "active",
                    expires_at=row["expires_at"],
                    created_at=row["created_at"],
                    triggered_at=row["triggered_at"],
                    trigger_price=float(row["current_value"]) if row["current_value"] else None,
                )
            )

        return alerts

    async def get_alert_stats(self, user_id: UUID) -> dict:
        """Get alert statistics for a user."""

        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_alerts,
                    COUNT(*) FILTER (WHERE is_active = true AND triggered_at IS NULL) as active_alerts,
                    COUNT(*) FILTER (WHERE triggered_at IS NOT NULL) as triggered_alerts,
                    COUNT(*) FILTER (WHERE expires_at < NOW()) as expired_alerts
                FROM price_alerts
                WHERE user_id = $1
            """,
                user_id,
            )

        return (
            dict(stats)
            if stats
            else {"total_alerts": 0, "active_alerts": 0, "triggered_alerts": 0, "expired_alerts": 0}
        )

    async def _load_active_alerts(self):
        """Load active alerts from database."""

        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM price_alerts
                WHERE is_active = true
                AND triggered_at IS NULL
                AND (expires_at IS NULL OR expires_at > NOW())
            """
            )

        self.active_alerts.clear()

        for row in rows:
            try:
                # Handle notification_methods (Postgres array)
                methods_list = row["notification_methods"] or []

                notification_methods = [
                    NotificationMethod(method)
                    for method in methods_list
                    if method in [m.value for m in NotificationMethod]
                ]

                # Map DB schema to PriceAlert model
                alert = PriceAlert(
                    id=row["id"],
                    user_id=row["user_id"],
                    symbol=row["symbol"],
                    alert_type=AlertType(row["alert_type"]),
                    condition_value=float(
                        row["target_value"]
                    ),  # Map target_value -> condition_value
                    condition_value2=None,
                    notification_methods=notification_methods,
                    message=None,
                    is_active=row["status"] == "active",  # Map status -> is_active
                    expires_at=row["expires_at"],
                    created_at=row["created_at"],
                )

                await self._add_to_monitoring(alert)

            except Exception as e:
                logger.error(f"Failed to load alert {row['id']}: {e}")

        logger.info(f"Loaded {len(self.active_alerts)} symbols with active alerts")

    async def _add_to_monitoring(self, alert: PriceAlert):
        """Add alert to active monitoring."""

        symbol = alert.symbol.upper()
        if symbol not in self.active_alerts:
            self.active_alerts[symbol] = []

        # Remove existing alert with same ID
        self.active_alerts[symbol] = [a for a in self.active_alerts[symbol] if a.id != alert.id]

        # Add new alert
        self.active_alerts[symbol].append(alert)

    async def _reload_symbol_alerts(self, symbol: str):
        """Reload alerts for a specific symbol."""

        symbol = symbol.upper()

        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM price_alerts
                WHERE symbol = $1 AND status = 'active'
                AND triggered_at IS NULL
                AND (expires_at IS NULL OR expires_at > NOW())
            """,
                symbol,
            )

        # Clear existing alerts for this symbol
        if symbol in self.active_alerts:
            del self.active_alerts[symbol]

        # Load fresh alerts
        for row in rows:
            try:
                # Handle notification_methods (Postgres array)
                methods_list = row["notification_methods"] or []

                notification_methods = [
                    NotificationMethod(method)
                    for method in methods_list
                    if method in [m.value for m in NotificationMethod]
                ]

                alert = PriceAlert(
                    id=row["id"],
                    user_id=row["user_id"],
                    symbol=row["symbol"],
                    alert_type=AlertType(row["alert_type"]),
                    condition_value=float(
                        row["target_value"]
                    ),  # Map target_value -> condition_value
                    condition_value2=None,
                    notification_methods=notification_methods,
                    message=None,
                    is_active=row["status"] == "active",
                    expires_at=row["expires_at"],
                    created_at=row["created_at"],
                )

                await self._add_to_monitoring(alert)

            except Exception as e:
                logger.error(f"Failed to reload alert {row['id']}: {e}")

    async def _check_alerts(self):
        """Check all active alerts against current market data."""

        if not self.active_alerts:
            return

        # Get current market data for all symbols
        symbols = list(self.active_alerts.keys())
        market_data = await self._get_market_data(symbols)

        triggered_alerts = []

        for symbol, alerts in self.active_alerts.items():
            if symbol not in market_data:
                continue

            price_data = market_data[symbol]
            current_price = float(price_data.get("price", 0))

            if current_price <= 0:
                continue

            for alert in alerts:
                try:
                    if await self._check_alert_condition(alert, price_data):
                        # Alert triggered!
                        trigger = AlertTrigger(
                            alert_id=alert.id,
                            symbol=symbol,
                            trigger_price=current_price,
                            trigger_time=datetime.utcnow(),
                            condition_met=self._get_condition_description(alert, current_price),
                            user_id=alert.user_id,
                            notification_methods=[
                                method.value for method in alert.notification_methods
                            ],
                            message=alert.message or f"{symbol} price alert triggered",
                        )

                        triggered_alerts.append(trigger)

                except Exception as e:
                    logger.error(f"Error checking alert {alert.id}: {e}")

        # Process triggered alerts
        for trigger in triggered_alerts:
            await self._process_triggered_alert(trigger)

    async def _get_market_data(self, symbols: list[str]) -> dict:
        """Get current market data for symbols."""

        # Mock implementation - replace with real market data
        market_data = {}

        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            for symbol in symbols:
                try:
                    # Get latest price from market_data_cache
                    row = await conn.fetchrow(
                        """
                        SELECT price, volume, bid, ask, change_percent
                        FROM market_data_cache
                        WHERE symbol = $1
                    """,
                        symbol,
                    )

                    if row:
                        market_data[symbol] = {
                            "price": float(row["price"]),
                            "volume": int(row["volume"] or 0),
                            "bid": float(row["bid"] or 0),
                            "ask": float(row["ask"] or 0),
                            "change_percent": float(row["change_percent"] or 0),
                        }

                except Exception as e:
                    logger.warning(f"Failed to get market data for {symbol}: {e}")

        return market_data

    async def _check_alert_condition(self, alert: PriceAlert, price_data: dict) -> bool:
        """Check if alert condition is met."""

        current_price = price_data["price"]

        if alert.alert_type == AlertType.PRICE_ABOVE:
            return current_price >= alert.condition_value

        elif alert.alert_type == AlertType.PRICE_BELOW:
            return current_price <= alert.condition_value

        elif alert.alert_type == AlertType.PRICE_CHANGE_PERCENT:
            change_percent = abs(price_data.get("change_percent", 0))
            return change_percent >= alert.condition_value

        elif alert.alert_type == AlertType.VOLUME_SPIKE:
            # Check if volume is X times higher than average
            volume = price_data.get("volume", 0)
            # Mock average volume check
            avg_volume = 1000000  # Should get from database
            return volume >= avg_volume * alert.condition_value

        elif alert.alert_type == AlertType.TECHNICAL_INDICATOR:
            # Placeholder for technical indicator alerts
            return False

        return False

    def _get_condition_description(self, alert: PriceAlert, trigger_price: float) -> str:
        """Get human-readable condition description."""

        if alert.alert_type == AlertType.PRICE_ABOVE:
            return f"Price ${trigger_price:.2f} above ${alert.condition_value:.2f}"

        elif alert.alert_type == AlertType.PRICE_BELOW:
            return f"Price ${trigger_price:.2f} below ${alert.condition_value:.2f}"

        elif alert.alert_type == AlertType.PRICE_CHANGE_PERCENT:
            return f"Price change {alert.condition_value:.1f}% threshold reached"

        elif alert.alert_type == AlertType.VOLUME_SPIKE:
            return f"Volume spike {alert.condition_value}x detected"

        return "Alert condition met"

    async def _process_triggered_alert(self, trigger: AlertTrigger):
        """Process a triggered alert."""

        logger.info(f"🚨 Alert triggered: {trigger.symbol} - {trigger.condition_met}")

        try:
            # Mark alert as triggered in database
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE price_alerts
                    SET triggered_at = $2, current_value = $3, status = 'triggered'
                    WHERE id = $1
                """,
                    trigger.alert_id,
                    trigger.trigger_time,
                    trigger.trigger_price,
                )

                # Store alert trigger history
                # NOTE: alert_triggers table does not exist yet, skipping insertion
                # await conn.execute("""
                #     INSERT INTO alert_triggers (
                #         id, alert_id, user_id, symbol, trigger_price, trigger_time,
                #         condition_met, notification_methods, message
                #     ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                # """,
                #     uuid4(), trigger.alert_id, trigger.user_id, trigger.symbol,
                #     trigger.trigger_price, trigger.trigger_time, trigger.condition_met,
                #     json.dumps(trigger.notification_methods), trigger.message
                # )

            # Remove from active monitoring
            if trigger.symbol in self.active_alerts:
                self.active_alerts[trigger.symbol] = [
                    alert
                    for alert in self.active_alerts[trigger.symbol]
                    if alert.id != trigger.alert_id
                ]

                if not self.active_alerts[trigger.symbol]:
                    del self.active_alerts[trigger.symbol]

            # Send notifications
            await self._send_notifications(trigger)

        except Exception as e:
            logger.error(f"Failed to process triggered alert {trigger.alert_id}: {e}")

    async def _send_notifications(self, trigger: AlertTrigger):
        """Send notifications for triggered alert."""

        # Mock notification sending - implement with real services
        for method in trigger.notification_methods:
            try:
                if method == "email":
                    await self._send_email_notification(trigger)
                elif method == "sms":
                    await self._send_sms_notification(trigger)
                elif method == "push":
                    await self._send_push_notification(trigger)
                elif method == "in_app":
                    await self._send_in_app_notification(trigger)

            except Exception as e:
                logger.error(f"Failed to send {method} notification: {e}")

    async def _send_email_notification(self, trigger: AlertTrigger):
        """Send email notification."""
        # TODO: Implement with SendGrid/AWS SES
        logger.info(f"📧 Email notification sent for {trigger.symbol} alert")

    async def _send_sms_notification(self, trigger: AlertTrigger):
        """Send SMS notification."""
        # TODO: Implement with Twilio
        logger.info(f"📱 SMS notification sent for {trigger.symbol} alert")

    async def _send_push_notification(self, trigger: AlertTrigger):
        """Send push notification."""
        # TODO: Implement with Firebase Cloud Messaging
        logger.info(f"🔔 Push notification sent for {trigger.symbol} alert")

    async def _send_in_app_notification(self, trigger: AlertTrigger):
        """Send in-app notification."""
        # Store in database for in-app display
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO notifications (
                    id, user_id, title, message, notification_type, metadata, created_at
                ) VALUES ($1, $2, $3, $4, 'alert', $5, $6)
            """,
                uuid4(),
                trigger.user_id,
                f"{trigger.symbol} Price Alert",
                f"{trigger.symbol}: {trigger.condition_met}",
                json.dumps(
                    {
                        "alert_id": str(trigger.alert_id),
                        "symbol": trigger.symbol,
                        "trigger_price": trigger.trigger_price,
                        "condition_met": trigger.condition_met,
                    }
                ),
                trigger.trigger_time,
            )

        logger.info(f"🔔 In-app notification created for {trigger.symbol} alert")

    async def _validate_alert(self, alert: PriceAlert):
        """Validate alert parameters."""

        if not alert.symbol:
            raise ValueError("Symbol is required")

        if alert.condition_value <= 0:
            raise ValueError("Condition value must be positive")

        if not alert.notification_methods:
            raise ValueError("At least one notification method is required")

        # Validate symbol exists
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            exists = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM market_data_cache WHERE symbol = $1
                )
            """,
                alert.symbol.upper(),
            )

            if not exists:
                raise ValueError(f"Symbol {alert.symbol} not found")


# Global alert service instance
_alert_service = None


def get_alert_service() -> PriceAlertService:
    """Get the global alert service instance."""
    global _alert_service
    if _alert_service is None:
        _alert_service = PriceAlertService()
    return _alert_service
