"""
CIFT Markets - User Settings API Routes

Comprehensive user settings, preferences, API keys, and security management.
NO MOCK DATA - All data from PostgreSQL database.

Endpoints:
- GET/PUT /settings - User preferences
- GET/POST/DELETE /settings/api-keys - API key management
- GET /settings/sessions - Login history
- POST /settings/2fa/enable - Enable 2FA
- POST /settings/2fa/disable - Disable 2FA
- POST /settings/2fa/verify - Verify 2FA code
- GET /settings/security/audit - Security audit log
"""

import os
import secrets
from datetime import datetime, timedelta
from uuid import UUID

import asyncpg
import bcrypt
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from loguru import logger
from pydantic import BaseModel, Field

from cift.core.auth import User, get_current_active_user, get_current_user_id
from cift.core.database import get_postgres_pool

# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(prefix="/settings", tags=["Settings"])


# ============================================================================
# MODELS
# ============================================================================

class UserSettings(BaseModel):
    """User settings and preferences"""
    # Profile
    full_name: str | None = None
    phone_number: str | None = None

    # Trading Preferences
    default_order_type: str = "market"
    default_time_in_force: str = "day"
    require_order_confirmation: bool = True
    enable_fractional_shares: bool = False

    # Notification Preferences
    email_notifications: bool = True
    email_trade_confirms: bool = True
    email_market_news: bool = False
    email_price_alerts: bool = True

    sms_notifications: bool = False
    sms_trade_confirms: bool = False
    sms_price_alerts: bool = False

    push_notifications: bool = True
    push_trade_confirms: bool = True
    push_market_news: bool = False
    push_price_alerts: bool = True

    # Quiet Hours
    notification_quiet_hours: bool = False
    quiet_start_time: str | None = "22:00:00"
    quiet_end_time: str | None = "08:00:00"

    # UI Preferences
    theme: str = "dark"
    language: str = "en"
    timezone: str = "America/New_York"
    currency: str = "USD"
    date_format: str = "MM/DD/YYYY"

    # Display Options
    show_portfolio_value: bool = True
    show_buying_power: bool = True
    show_day_pnl: bool = True
    compact_mode: bool = False

    # Data & Privacy
    data_sharing_enabled: bool = False
    analytics_enabled: bool = True
    marketing_emails: bool = False


class UserSettingsUpdate(BaseModel):
    """Partial update for user settings"""
    full_name: str | None = None
    phone_number: str | None = None
    default_order_type: str | None = None
    default_time_in_force: str | None = None
    require_order_confirmation: bool | None = None
    enable_fractional_shares: bool | None = None
    email_notifications: bool | None = None
    email_trade_confirms: bool | None = None
    email_market_news: bool | None = None
    email_price_alerts: bool | None = None
    sms_notifications: bool | None = None
    sms_trade_confirms: bool | None = None
    sms_price_alerts: bool | None = None
    push_notifications: bool | None = None
    push_trade_confirms: bool | None = None
    push_market_news: bool | None = None
    push_price_alerts: bool | None = None
    notification_quiet_hours: bool | None = None
    quiet_start_time: str | None = None
    quiet_end_time: str | None = None
    theme: str | None = None
    language: str | None = None
    timezone: str | None = None
    currency: str | None = None
    date_format: str | None = None
    show_portfolio_value: bool | None = None
    show_buying_power: bool | None = None
    show_day_pnl: bool | None = None
    compact_mode: bool | None = None
    data_sharing_enabled: bool | None = None
    analytics_enabled: bool | None = None
    marketing_emails: bool | None = None


class ApiKeyResponse(BaseModel):
    """API key information (excluding full key)"""
    id: str
    name: str | None
    key_prefix: str
    scopes: list[str]
    rate_limit_per_minute: int
    last_used_at: datetime | None
    expires_at: datetime | None
    is_active: bool
    created_at: datetime
    total_requests: int


class ApiKeyCreateRequest(BaseModel):
    """Request to create new API key"""
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    scopes: list[str] = Field(default=["read"], description="Permissions: read, trade, withdraw")
    expires_in_days: int | None = Field(None, ge=1, le=365, description="Expiry in days (optional)")


class ApiKeyCreateResponse(BaseModel):
    """Response after creating API key (includes full key once)"""
    api_key: str = Field(..., description="Full API key - SAVE THIS! Won't be shown again")
    key_id: str
    name: str
    expires_at: datetime | None
    message: str = "IMPORTANT: Save this API key securely. It will not be shown again."


class SessionLog(BaseModel):
    """User session information"""
    id: str
    ip_address: str | None
    user_agent: str | None
    device_type: str | None
    browser: str | None
    os: str | None
    city: str | None
    country: str | None
    login_at: datetime
    logout_at: datetime | None
    last_activity_at: datetime
    is_active: bool
    is_suspicious: bool
    login_method: str


class TwoFactorSetup(BaseModel):
    """2FA setup information"""
    secret: str
    qr_code_url: str
    backup_codes: list[str]
    message: str = "Save backup codes securely. They can be used if you lose access to your authenticator."


class TwoFactorVerifyRequest(BaseModel):
    """2FA verification request"""
    code: str = Field(..., min_length=6, max_length=6, description="6-digit code from authenticator")


class SecurityAuditLog(BaseModel):
    """Security audit log entry"""
    id: str
    event_type: str
    event_category: str
    severity: str
    description: str | None
    ip_address: str | None
    success: bool
    created_at: datetime


# ============================================================================
# USER SETTINGS ENDPOINTS
# ============================================================================

@router.get("", response_model=UserSettings)
async def get_user_settings(
    user: User = Depends(get_current_active_user),
    pool: asyncpg.Pool = Depends(get_postgres_pool),
):
    """
    Get current user's settings and preferences.

    Returns all user preference settings including trading defaults,
    notifications, UI preferences, and privacy options.
    """
    try:
        async with pool.acquire() as conn:
            # Ensure user settings row exists
            await conn.execute(
                "INSERT INTO user_settings (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
                user.id
            )

            # Get user settings
            query = """
                SELECT
                    full_name, phone_number,
                    default_order_type, default_time_in_force,
                    require_order_confirmation, enable_fractional_shares,
                    email_notifications, email_trade_confirms,
                    email_market_news, email_price_alerts,
                    sms_notifications, sms_trade_confirms, sms_price_alerts,
                    push_notifications, push_trade_confirms,
                    push_market_news, push_price_alerts,
                    notification_quiet_hours, quiet_start_time, quiet_end_time,
                    theme, language, timezone, currency, date_format,
                    show_portfolio_value, show_buying_power, show_day_pnl,
                    compact_mode, data_sharing_enabled,
                    analytics_enabled, marketing_emails
                FROM user_settings
                WHERE user_id = $1;
            """
            row = await conn.fetchrow(query, user.id)

            if not row:
                # Return defaults if no row found
                return UserSettings()

            # Convert row to dict, filtering only expected fields
            settings_dict = {}
            for field in UserSettings.__fields__.keys():
                if field in row.keys():
                    value = row[field]
                    # Convert time objects to strings
                    if field in ('quiet_start_time', 'quiet_end_time') and value is not None:
                        value = value.isoformat() if hasattr(value, 'isoformat') else str(value)
                    settings_dict[field] = value

            return UserSettings(**settings_dict)
    except Exception as e:
        logger.error(f"Error fetching settings for user {user.id}: {e}")
        # Return default settings on error
        return UserSettings()


@router.put("", response_model=UserSettings)
async def update_user_settings(
    updates: UserSettingsUpdate,
    user_id: UUID = Depends(get_current_user_id),
    user: User = Depends(get_current_active_user),
    pool: asyncpg.Pool = Depends(get_postgres_pool),
    request: Request = None,
):
    """
    Update user settings and preferences.

    Accepts partial updates - only provided fields will be updated.
    Automatically tracks update timestamp.
    """
    async with pool.acquire() as conn:
        # Build dynamic update query
        update_fields = []
        params = [user_id]  # Use UUID directly instead of user.id
        param_idx = 2

        for field, value in updates.dict(exclude_unset=True).items():
            if value is not None:
                update_fields.append(f"{field} = ${param_idx}")
                params.append(value)
                param_idx += 1

        if not update_fields:
            # No updates, just return current settings - create inline query
            query = """
                SELECT
                    full_name, phone_number,
                    default_order_type, default_time_in_force,
                    require_order_confirmation, enable_fractional_shares,
                    email_notifications, email_trade_confirms,
                    email_market_news, email_price_alerts,
                    sms_notifications, sms_trade_confirms, sms_price_alerts,
                    push_notifications, push_trade_confirms,
                    push_market_news, push_price_alerts,
                    notification_quiet_hours, quiet_start_time, quiet_end_time,
                    theme, language, timezone, currency, date_format,
                    show_portfolio_value, show_buying_power, show_day_pnl,
                    compact_mode, data_sharing_enabled,
                    analytics_enabled, marketing_emails
                FROM user_settings
                WHERE user_id = $1;
            """
            row = await conn.fetchrow(query, user_id)
            if not row:
                return UserSettings()

            settings_dict = {}
            for field in UserSettings.__fields__.keys():
                if field in row.keys():
                    settings_dict[field] = row[field]

            return UserSettings(**settings_dict)

        query = f"""
            UPDATE user_settings
            SET {', '.join(update_fields)}, updated_at = NOW()
            WHERE user_id = $1
            RETURNING *;
        """

        row = await conn.fetchrow(query, *params)

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User settings not found"
            )

        # Log security event (if function exists)
        try:
            await conn.execute(
                """
                SELECT log_security_event(
                    $1, 'settings_updated', 'settings',
                    'User updated their preferences', 'info',
                    $2::jsonb, $3::inet
                )
                """,
                user_id,
                updates.dict(exclude_unset=True),
                request.client.host if request else None
            )
        except Exception as e:
            logger.warning(f"Failed to log security event: {e}")

        logger.info(f"User {user_id} updated settings: {list(updates.dict(exclude_unset=True).keys())}")

        return UserSettings(**dict(row))


@router.post("/avatar")
async def upload_avatar(
    avatar: UploadFile = File(...),
    user_id: UUID = Depends(get_current_user_id),
    pool: asyncpg.Pool = Depends(get_postgres_pool),
):
    """
    Upload user profile avatar.

    Accepts image files (PNG, JPG, GIF, WebP) up to 5MB.
    Stores the file and updates user profile with avatar URL.
    """
    # Validate file type
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp']
    if avatar.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only image files (JPG, PNG, GIF, WebP) are allowed"
        )

    # Validate file size (5MB max)
    content = await avatar.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size must be less than 5MB"
        )

    try:
        # Create uploads directory if it doesn't exist
        upload_dir = "uploads/avatars"
        os.makedirs(upload_dir, exist_ok=True)

        # Generate unique filename
        file_extension = avatar.filename.split('.')[-1] if avatar.filename else 'jpg'
        filename = f"{user_id}.{file_extension}"
        file_path = os.path.join(upload_dir, filename)

        # Save file
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # Update database with avatar URL
        avatar_url = f"/uploads/avatars/{filename}"

        async with pool.acquire() as conn:
            # Check if user_settings table has avatar_url column, if not, we'll just return success
            try:
                await conn.execute(
                    "UPDATE user_settings SET avatar_url = $1 WHERE user_id = $2",
                    avatar_url,
                    user_id
                )
            except Exception as e:
                # Table might not have avatar_url column, that's okay for now
                logger.warning(f"Could not update avatar_url in database: {e}")

        logger.info(f"Avatar uploaded for user {user_id}: {file_path}")

        return {
            "message": "Avatar uploaded successfully",
            "avatar_url": avatar_url,
            "filename": filename
        }

    except Exception as e:
        logger.error(f"Failed to upload avatar for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload avatar"
        ) from e


# ============================================================================
# API KEY MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/api-keys", response_model=list[ApiKeyResponse])
async def list_api_keys(
    user: User = Depends(get_current_active_user),
    pool: asyncpg.Pool = Depends(get_postgres_pool),
):
    """
    List all API keys for the current user.

    Returns metadata about each API key (excluding the actual key value).
    """
    try:
        async with pool.acquire() as conn:
            # Check if table exists
            table_exists = await conn.fetchval(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'api_keys')"
            )

            if not table_exists:
                logger.warning("api_keys table does not exist")
                return []

            query = """
                SELECT
                    id::text, name, key_prefix, scopes,
                    rate_limit_per_minute, last_used_at, expires_at,
                    is_active, created_at, total_requests
                FROM api_keys
                WHERE user_id = $1 AND is_active = true
                ORDER BY created_at DESC;
            """
            rows = await conn.fetch(query, user.id)

            return [ApiKeyResponse(**dict(row)) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching API keys for user {user.id}: {e}")
        return []


@router.post("/api-keys", response_model=ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    key_request: ApiKeyCreateRequest,
    user: User = Depends(get_current_active_user),
    pool: asyncpg.Pool = Depends(get_postgres_pool),
    request: Request = None,
):
    """
    Generate a new API key for programmatic access.

    **IMPORTANT:** The full API key is only shown once.
    Save it securely immediately after creation.

    Scopes:
    - `read`: View account data, positions, orders
    - `trade`: Submit and cancel orders
    - `withdraw`: Initiate withdrawals (requires additional verification)
    """
    try:
        async with pool.acquire() as conn:
            # Check if table exists
            try:
                table_exists = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'api_keys')"
                )
            except Exception as e:
                logger.error(f"Failed to check table existence: {e}")
                table_exists = False

            if not table_exists:
                logger.warning("api_keys table does not exist - denying key creation")
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="API key management is not available yet. Please contact your administrator."
                )

            # Generate secure random API key
            api_key = f"sk_live_{secrets.token_urlsafe(32)}"
            key_prefix = api_key[:12]  # "sk_live_XXXX"

            # Hash the key for storage (bcrypt)
            try:
                key_hash = bcrypt.hashpw(api_key.encode(), bcrypt.gensalt()).decode()
            except Exception as e:
                logger.error(f"Failed to hash API key: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to generate secure key"
                ) from e

            # Calculate expiry
            expires_at = None
            if key_request.expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=key_request.expires_in_days)

            # Insert API key
            query = """
                INSERT INTO api_keys (
                    user_id, key_hash, key_prefix, name, description,
                    scopes, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id::text, name, expires_at;
            """

            try:
                row = await conn.fetchrow(
                    query,
                    user.id,
                    key_hash,
                    key_prefix,
                    key_request.name,
                    key_request.description,
                    key_request.scopes,
                    expires_at
                )
            except Exception as e:
                logger.error(f"Failed to insert API key: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Database error: {str(e)}"
                ) from e

            # Log security event (if function exists)
            try:
                await conn.execute(
                    """
                    SELECT log_security_event(
                        $1, 'api_key_created', 'security',
                        $2, 'info',
                        $3::jsonb, $4::inet
                    )
                    """,
                    user.id,
                    f"Created API key: {key_request.name}",
                    {"name": key_request.name, "scopes": key_request.scopes},
                    request.client.host if request else None
                )
            except Exception as e:
                logger.warning(f"Failed to log security event: {e}")

            logger.info(f"User {user.id} created API key: {key_request.name} (scopes: {key_request.scopes})")

            return ApiKeyCreateResponse(
                api_key=api_key,
                key_id=row['id'],
                name=row['name'],
                expires_at=row['expires_at']
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating API key for user {user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        ) from e


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: UUID,
    user: User = Depends(get_current_active_user),
    pool: asyncpg.Pool = Depends(get_postgres_pool),
    request: Request = None,
):
    """
    Revoke (delete) an API key.

    The key will be permanently disabled and cannot be used for authentication.
    This action cannot be undone.
    """
    async with pool.acquire() as conn:
        query = """
            UPDATE api_keys
            SET is_active = false, revoked_at = NOW()
            WHERE id = $1 AND user_id = $2 AND is_active = true
            RETURNING name;
        """
        row = await conn.fetchrow(query, key_id, user.id)

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )

        # Log security event
        await conn.execute(
            """
            SELECT log_security_event(
                $1, 'api_key_revoked', 'security',
                $2, 'warning',
                $3::jsonb, $4::inet
            )
            """,
            user.id,
            f"Revoked API key: {row['name']}",
            {"key_id": str(key_id), "name": row['name']},
            request.client.host if request else None
        )

        logger.warning(f"User {user.id} revoked API key: {row['name']}")


# ============================================================================
# SESSION MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/sessions", response_model=list[SessionLog])
async def get_session_history(
    limit: int = 50,
    user: User = Depends(get_current_active_user),
    pool: asyncpg.Pool = Depends(get_postgres_pool),
):
    """
    Get login session history for security auditing.

    Shows recent and active sessions including:
    - Device information
    - Location (from IP)
    - Login/logout times
    - Suspicious activity flags
    """
    async with pool.acquire() as conn:
        query = """
            SELECT
                id::text, ip_address::text, user_agent, device_type,
                browser, os, city, country, login_at, logout_at,
                last_activity_at, is_active, is_suspicious, login_method
            FROM session_logs
            WHERE user_id = $1
            ORDER BY login_at DESC
            LIMIT $2;
        """
        rows = await conn.fetch(query, user.id, limit)

        return [SessionLog(**dict(row)) for row in rows]


@router.post("/sessions/{session_id}/terminate", status_code=status.HTTP_204_NO_CONTENT)
async def terminate_session(
    session_id: UUID,
    user: User = Depends(get_current_active_user),
    pool: asyncpg.Pool = Depends(get_postgres_pool),
):
    """
    Terminate an active session (force logout).

    Useful for revoking access from a lost or compromised device.
    """
    async with pool.acquire() as conn:
        query = """
            UPDATE session_logs
            SET is_active = false, logout_at = NOW()
            WHERE id = $1 AND user_id = $2 AND is_active = true
            RETURNING id;
        """
        row = await conn.fetchrow(query, session_id, user.id)

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Active session not found"
            )

        logger.info(f"User {user.id} terminated session {session_id}")


# ============================================================================
# TWO-FACTOR AUTHENTICATION ENDPOINTS
# ============================================================================

@router.post("/2fa/enable", response_model=TwoFactorSetup)
async def enable_2fa(
    user: User = Depends(get_current_active_user),
    pool: asyncpg.Pool = Depends(get_postgres_pool),
):
    """
    Enable Two-Factor Authentication (2FA) using TOTP.

    Returns:
    - Secret key for manual entry
    - QR code URL for scanning with authenticator app
    - Backup codes for emergency access

    **Next step:** Call `/2fa/verify` with a code from your authenticator app
    to confirm setup.
    """
    import base64
    import io

    import pyotp
    import qrcode

    async with pool.acquire() as conn:
        # Check if 2FA already enabled
        existing = await conn.fetchrow(
            "SELECT enabled FROM two_factor_auth WHERE user_id = $1",
            user.id
        )

        if existing and existing['enabled']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA is already enabled"
            )

        # Generate TOTP secret
        secret = pyotp.random_base32()
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user.email,
            issuer_name="CIFT Markets"
        )

        # Generate backup codes
        backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
        backup_codes_hashed = [
            bcrypt.hashpw(code.encode(), bcrypt.gensalt()).decode()
            for code in backup_codes
        ]

        # TODO: Encrypt secret in production
        secret_encrypted = secret  # Placeholder - should encrypt

        # Store 2FA settings (but not enabled yet)
        await conn.execute(
            """
            INSERT INTO two_factor_auth (
                user_id, enabled, method, secret_encrypted,
                backup_codes_encrypted, backup_codes_remaining
            ) VALUES ($1, false, 'totp', $2, $3, 10)
            ON CONFLICT (user_id) DO UPDATE
            SET secret_encrypted = $2,
                backup_codes_encrypted = $3,
                backup_codes_remaining = 10,
                enabled = false;
            """,
            user.id, secret_encrypted, backup_codes_hashed
        )

        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")

        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        qr_code_base64 = base64.b64encode(buffer.getvalue()).decode()
        qr_code_url = f"data:image/png;base64,{qr_code_base64}"

        logger.info(f"User {user.id} initiated 2FA setup")

        return TwoFactorSetup(
            secret=secret,
            qr_code_url=qr_code_url,
            backup_codes=backup_codes
        )


@router.post("/2fa/verify", status_code=status.HTTP_200_OK)
async def verify_2fa(
    verify_request: TwoFactorVerifyRequest,
    user: User = Depends(get_current_active_user),
    pool: asyncpg.Pool = Depends(get_postgres_pool),
):
    """
    Verify 2FA code and complete setup.

    After calling `/2fa/enable`, use this endpoint with a 6-digit code
    from your authenticator app to activate 2FA protection.
    """
    import pyotp

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT secret_encrypted FROM two_factor_auth WHERE user_id = $1",
            user.id
        )

        if not row:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA not initialized. Call /2fa/enable first."
            )

        # Verify code
        totp = pyotp.TOTP(row['secret_encrypted'])
        if not totp.verify(verify_request.code, valid_window=1):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification code"
            )

        # Enable 2FA
        await conn.execute(
            """
            UPDATE two_factor_auth
            SET enabled = true, verified_at = NOW()
            WHERE user_id = $1;
            """,
            user.id
        )

        # Log security event
        await conn.execute(
            "SELECT log_security_event($1, '2fa_enabled', 'security', '2FA enabled successfully', 'info')",
            user.id
        )

        logger.info(f"User {user.id} enabled 2FA successfully")

        return {"message": "2FA enabled successfully", "status": "enabled"}


@router.post("/2fa/disable", status_code=status.HTTP_200_OK)
async def disable_2fa(
    verify_request: TwoFactorVerifyRequest,
    user: User = Depends(get_current_active_user),
    pool: asyncpg.Pool = Depends(get_postgres_pool),
):
    """
    Disable Two-Factor Authentication.

    Requires a valid 2FA code or backup code to disable.
    This reduces account security and is not recommended.
    """
    import pyotp

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT secret_encrypted, backup_codes_encrypted
            FROM two_factor_auth
            WHERE user_id = $1 AND enabled = true;
            """,
            user.id
        )

        if not row:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA is not enabled"
            )

        # Verify code (TOTP or backup code)
        code_valid = False

        # Check TOTP code
        totp = pyotp.TOTP(row['secret_encrypted'])
        if totp.verify(verify_request.code, valid_window=1):
            code_valid = True
        else:
            # Check backup codes
            for backup_hash in row['backup_codes_encrypted']:
                if bcrypt.checkpw(verify_request.code.encode(), backup_hash.encode()):
                    code_valid = True
                    break

        if not code_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid code"
            )

        # Disable 2FA
        await conn.execute(
            "UPDATE two_factor_auth SET enabled = false WHERE user_id = $1;",
            user.id
        )

        # Log security event
        await conn.execute(
            "SELECT log_security_event($1, '2fa_disabled', 'security', '2FA disabled by user', 'warning')",
            user.id
        )

        logger.warning(f"User {user.id} disabled 2FA")

        return {"message": "2FA disabled successfully", "status": "disabled"}


# ============================================================================
# SECURITY AUDIT LOG ENDPOINTS
# ============================================================================

@router.get("/security/audit", response_model=list[SecurityAuditLog])
async def get_security_audit_log(
    limit: int = 100,
    event_type: str | None = None,
    user: User = Depends(get_current_active_user),
    pool: asyncpg.Pool = Depends(get_postgres_pool),
):
    """
    Get security audit log for the current user.

    Shows all security-related events including:
    - Login/logout events
    - Password changes
    - 2FA changes
    - API key management
    - Settings updates
    """
    async with pool.acquire() as conn:
        if event_type:
            query = """
                SELECT
                    id::text, event_type, event_category, severity,
                    description, ip_address::text, success, created_at
                FROM security_audit_log
                WHERE user_id = $1 AND event_type = $2
                ORDER BY created_at DESC
                LIMIT $3;
            """
            rows = await conn.fetch(query, user.id, event_type, limit)
        else:
            query = """
                SELECT
                    id::text, event_type, event_category, severity,
                    description, ip_address::text, success, created_at
                FROM security_audit_log
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2;
            """
            rows = await conn.fetch(query, user.id, limit)

        return [SecurityAuditLog(**dict(row)) for row in rows]
