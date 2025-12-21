"""
SMS SERVICE - RULES COMPLIANT
Sends SMS for phone verification, 2FA, and transaction notifications.
Uses Twilio, AWS SNS, or Africa's Talking for delivery.
"""

import secrets
from datetime import datetime, timedelta

from cift.core.config import settings
from cift.core.database import get_postgres_pool
from cift.core.logging import logger


class SMSService:
    """
    SMS service for sending verification codes and notifications
    Supports Twilio, AWS SNS, and Africa's Talking
    """

    def __init__(self):
        self.provider = getattr(settings, 'SMS_PROVIDER', 'twilio')  # 'twilio', 'aws_sns', 'africas_talking'
        self.from_number = getattr(settings, 'SMS_FROM_NUMBER', '+1234567890')

        # Twilio config
        self.twilio_sid = getattr(settings, 'TWILIO_ACCOUNT_SID', None)
        self.twilio_token = getattr(settings, 'TWILIO_AUTH_TOKEN', None)

        # Africa's Talking config
        self.at_username = getattr(settings, 'AFRICAS_TALKING_USERNAME', None)
        self.at_api_key = getattr(settings, 'AFRICAS_TALKING_API_KEY', None)

    async def send_sms(self, phone: str, message: str) -> bool:
        """
        Send SMS via configured provider

        Args:
            phone: Phone number in E.164 format (+1234567890)
            message: SMS message text

        Returns:
            True if sent successfully
        """
        if not phone.startswith('+'):
            logger.warning(f"Phone number must be in E.164 format: {phone}")
            return False

        try:
            if self.provider == 'twilio':
                return await self._send_via_twilio(phone, message)
            elif self.provider == 'africas_talking':
                return await self._send_via_africas_talking(phone, message)
            elif self.provider == 'aws_sns':
                return await self._send_via_aws_sns(phone, message)
            else:
                logger.warning("SMS provider not configured, skipping SMS send")
                logger.info(f"Would send SMS to {phone}: {message}")
                return False

        except Exception as e:
            logger.error(f"Failed to send SMS to {phone}: {str(e)}")
            return False

    async def _send_via_twilio(self, phone: str, message: str) -> bool:
        """Send SMS via Twilio"""
        if not self.twilio_sid or not self.twilio_token:
            logger.warning("Twilio credentials not configured")
            logger.info(f"Would send Twilio SMS to {phone}: {message}")
            return False

        try:
            from twilio.rest import Client

            client = Client(self.twilio_sid, self.twilio_token)
            message = client.messages.create(
                body=message,
                from_=self.from_number,
                to=phone
            )

            logger.info(f"SMS sent via Twilio to {phone}: {message.sid}")
            return True

        except Exception as e:
            logger.error(f"Twilio SMS failed: {str(e)}")
            return False

    async def _send_via_africas_talking(self, phone: str, message: str) -> bool:
        """Send SMS via Africa's Talking (good for African numbers)"""
        if not self.at_username or not self.at_api_key:
            logger.warning("Africa's Talking credentials not configured")
            logger.info(f"Would send AT SMS to {phone}: {message}")
            return False

        try:
            import africastalking

            africastalking.initialize(self.at_username, self.at_api_key)
            sms = africastalking.SMS

            sms.send(message, [phone])

            logger.info(f"SMS sent via Africa's Talking to {phone}")
            return True

        except Exception as e:
            logger.error(f"Africa's Talking SMS failed: {str(e)}")
            return False

    async def _send_via_aws_sns(self, phone: str, message: str) -> bool:
        """Send SMS via AWS SNS"""
        try:
            import boto3

            client = boto3.client('sns', region_name='us-east-1')

            response = client.publish(
                PhoneNumber=phone,
                Message=message,
                MessageAttributes={
                    'AWS.SNS.SMS.SMSType': {
                        'DataType': 'String',
                        'StringValue': 'Transactional'
                    }
                }
            )

            logger.info(f"SMS sent via AWS SNS to {phone}: {response['MessageId']}")
            return True

        except Exception as e:
            logger.error(f"AWS SNS SMS failed: {str(e)}")
            return False

    async def send_verification_code(
        self,
        phone: str,
        purpose: str = 'phone_verification'
    ) -> str | None:
        """
        Send verification code to phone

        Args:
            phone: Phone number in E.164 format
            purpose: Purpose of verification ('phone_verification', '2fa', etc.)

        Returns:
            Verification code if sent successfully, None otherwise
        """
        # Generate 6-digit code
        code = ''.join([str(secrets.randbelow(10)) for _ in range(6)])

        # Store code in database
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO verification_codes (
                    phone,
                    code,
                    purpose,
                    expires_at,
                    created_at
                )
                VALUES ($1, $2, $3, $4, NOW())
                """,
                phone,
                code,
                purpose,
                datetime.utcnow() + timedelta(minutes=10),  # Expires in 10 minutes
            )

        # Send SMS
        message = f"Your CIFT Markets verification code is: {code}\n\nValid for 10 minutes.\n\nDo not share this code."

        success = await self.send_sms(phone, message)

        return code if success else None

    async def verify_code(
        self,
        phone: str,
        code: str,
        purpose: str = 'phone_verification'
    ) -> bool:
        """
        Verify code entered by user

        Args:
            phone: Phone number
            code: Code entered by user
            purpose: Purpose of verification

        Returns:
            True if code is valid
        """
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            # Check if code exists and is not expired
            result = await conn.fetchrow(
                """
                SELECT id, code, expires_at, verified_at
                FROM verification_codes
                WHERE phone = $1
                AND code = $2
                AND purpose = $3
                AND expires_at > NOW()
                AND verified_at IS NULL
                ORDER BY created_at DESC
                LIMIT 1
                """,
                phone,
                code,
                purpose,
            )

            if not result:
                return False

            # Mark as verified
            await conn.execute(
                """
                UPDATE verification_codes
                SET verified_at = NOW()
                WHERE id = $1
                """,
                result['id'],
            )

            return True

    async def send_transaction_completed(
        self,
        phone: str,
        amount: float,
        receipt: str | None = None
    ):
        """Send transaction completed notification via SMS"""
        message = f"CIFT Markets: Your transaction of ${amount:,.2f} has been completed successfully."

        if receipt:
            message += f" Receipt: {receipt}"

        await self.send_sms(phone, message)

    async def send_transaction_alert(
        self,
        phone: str,
        transaction_type: str,
        amount: float
    ):
        """Send transaction alert via SMS"""
        action = "deposit" if transaction_type == 'deposit' else "withdrawal"
        message = f"CIFT Markets: New {action} of ${amount:,.2f} initiated on your account. If this wasn't you, contact support immediately."

        await self.send_sms(phone, message)

    async def send_2fa_code(self, phone: str) -> str | None:
        """Send 2FA code for login"""
        return await self.send_verification_code(phone, '2fa')

    async def send_mpesa_verification(
        self,
        phone: str,
        amount: str = "1.00"
    ):
        """Send M-Pesa verification notification"""
        message = f"CIFT Markets: Check your M-Pesa phone for a verification request of KES {amount}. Enter your PIN to verify."

        await self.send_sms(phone, message)

    async def send_account_security_alert(
        self,
        phone: str,
        alert_type: str,
        details: str
    ):
        """Send security alert via SMS"""
        message = f"CIFT Markets SECURITY ALERT: {alert_type}. {details}. If this wasn't you, secure your account immediately."

        await self.send_sms(phone, message)


# Singleton instance
sms_service = SMSService()
