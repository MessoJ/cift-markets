"""
EMAIL SERVICE - RULES COMPLIANT
Sends transactional emails for verifications, notifications, and alerts.
Uses SMTP or SendGrid/AWS SES for production.
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from cift.core.config import settings
from cift.core.logging import logger


class EmailService:
    """
    Email service for sending transactional emails
    Supports SMTP and SendGrid/AWS SES
    """

    def __init__(self):
        self.smtp_host = getattr(settings, "SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = getattr(settings, "SMTP_PORT", 587)
        self.smtp_user = getattr(settings, "SMTP_USER", None)
        self.smtp_password = getattr(settings, "SMTP_PASSWORD", None)
        self.from_email = getattr(settings, "FROM_EMAIL", "noreply@ciftmarkets.com")
        self.from_name = getattr(settings, "FROM_NAME", "CIFT Markets")

    async def send_email(
        self, to_email: str, subject: str, html_body: str, text_body: str | None = None
    ) -> bool:
        """
        Send email via SMTP

        Args:
            to_email: Recipient email address
            subject: Email subject
            html_body: HTML email body
            text_body: Plain text fallback

        Returns:
            True if sent successfully
        """
        if not self.smtp_user or not self.smtp_password:
            logger.warning("SMTP credentials not configured, skipping email send")
            logger.info(f"Would send email to {to_email}: {subject}")
            return False

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = to_email
            msg["Subject"] = subject

            # Add text and HTML parts
            if text_body:
                msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            # Send via SMTP
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent to {to_email}: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {str(e)}")
            return False

    async def send_payment_method_verification(
        self,
        email: str,
        payment_method_type: str,
        verification_type: str,
        verification_details: dict,
    ):
        """
        Send payment method verification instructions

        Args:
            email: User email
            payment_method_type: Type of payment method
            verification_type: Type of verification (micro_deposit, stk_push, oauth)
            verification_details: Additional verification info
        """
        subject = f"Verify Your {payment_method_type.replace('_', ' ').title()}"

        if verification_type == "micro_deposit":
            html_body = """
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: #1a1d2e; color: white; padding: 20px; border-radius: 8px;">
                    <h2 style="color: #00d4ff;">Verify Your Bank Account</h2>
                    <p>We've sent two small deposits to your bank account. This process may take 1-3 business days.</p>
                    <p>Once you see these deposits in your account:</p>
                    <ol>
                        <li>Log in to CIFT Markets</li>
                        <li>Go to Funding → Payment Methods</li>
                        <li>Click "Verify" on your bank account</li>
                        <li>Enter the two deposit amounts</li>
                    </ol>
                    <p style="color: #fbbf24; margin-top: 20px;">
                        ⚠️ You have 3 days to complete verification.
                    </p>
                    <p style="margin-top: 30px; font-size: 12px; color: #9ca3af;">
                        If you didn't add this payment method, please contact support immediately.
                    </p>
                </div>
            </body>
            </html>
            """

        elif verification_type == "stk_push":
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: #1a1d2e; color: white; padding: 20px; border-radius: 8px;">
                    <h2 style="color: #00d64f;">Verify Your M-Pesa Account</h2>
                    <p>A verification request has been sent to your M-Pesa phone: {verification_details.get('phone', 'your phone')}</p>
                    <p>To complete verification:</p>
                    <ol>
                        <li>Check your phone for the M-Pesa STK push prompt</li>
                        <li>Enter your M-Pesa PIN to authorize</li>
                        <li>Confirm in the CIFT Markets app</li>
                    </ol>
                    <p style="color: #fbbf24; margin-top: 20px;">
                        ⚠️ This request expires in 5 minutes.
                    </p>
                </div>
            </body>
            </html>
            """

        elif verification_type == "oauth":
            oauth_url = verification_details.get("oauth_url", "")
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: #1a1d2e; color: white; padding: 20px; border-radius: 8px;">
                    <h2 style="color: #00d4ff;">Authorize Your {payment_method_type.title()} Account</h2>
                    <p>To complete verification, you need to authorize CIFT Markets to connect to your {payment_method_type.title()} account.</p>
                    <div style="margin: 30px 0;">
                        <a href="{oauth_url}" style="background: #00d4ff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">
                            Authorize {payment_method_type.title()}
                        </a>
                    </div>
                    <p style="color: #fbbf24;">
                        ⚠️ This link expires in 15 minutes.
                    </p>
                </div>
            </body>
            </html>
            """

        else:
            html_body = """
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: #1a1d2e; color: white; padding: 20px; border-radius: 8px;">
                    <h2 style="color: #00d4ff;">Verify Your Payment Method</h2>
                    <p>Please complete the verification process in the CIFT Markets app.</p>
                </div>
            </body>
            </html>
            """

        await self.send_email(email, subject, html_body)

    async def send_payment_method_verified(self, email: str, payment_method_type: str):
        """Send notification that payment method was verified"""
        subject = f"{payment_method_type.replace('_', ' ').title()} Verified Successfully"

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: #1a1d2e; color: white; padding: 20px; border-radius: 8px;">
                <h2 style="color: #10b981;">✓ Payment Method Verified</h2>
                <p>Your {payment_method_type.replace('_', ' ')} has been successfully verified and is now ready to use.</p>
                <p>You can now:</p>
                <ul>
                    <li>Deposit funds instantly</li>
                    <li>Withdraw profits</li>
                    <li>Set this as your default payment method</li>
                </ul>
                <div style="margin: 30px 0;">
                    <a href="https://app.ciftmarkets.com/funding" style="background: #00d4ff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">
                        Start Trading
                    </a>
                </div>
            </div>
        </body>
        </html>
        """

        await self.send_email(email, subject, html_body)

    async def send_transaction_completed(
        self, email: str, transaction_type: str, amount: float, transaction_id: str
    ):
        """Send notification that transaction completed"""
        action = "Deposit" if transaction_type == "deposit" else "Withdrawal"
        subject = f"{action} Completed - ${amount:,.2f}"

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: #1a1d2e; color: white; padding: 20px; border-radius: 8px;">
                <h2 style="color: #10b981;">✓ {action} Completed</h2>
                <div style="background: #2d3748; padding: 15px; border-radius: 6px; margin: 20px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="color: #9ca3af;">Amount:</span>
                        <span style="font-size: 20px; font-weight: bold;">${amount:,.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #9ca3af;">Transaction ID:</span>
                        <span style="font-family: monospace; font-size: 11px;">{transaction_id}</span>
                    </div>
                </div>
                <p>Your {transaction_type} has been processed successfully and your account balance has been updated.</p>
                <div style="margin: 30px 0;">
                    <a href="https://app.ciftmarkets.com/dashboard" style="background: #00d4ff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">
                        View Dashboard
                    </a>
                </div>
            </div>
        </body>
        </html>
        """

        await self.send_email(email, subject, html_body)

    async def send_transaction_failed(
        self, email: str, transaction_type: str, amount: float, reason: str
    ):
        """Send notification that transaction failed"""
        action = "Deposit" if transaction_type == "deposit" else "Withdrawal"
        subject = f"{action} Failed - ${amount:,.2f}"

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: #1a1d2e; color: white; padding: 20px; border-radius: 8px;">
                <h2 style="color: #ef4444;">✗ {action} Failed</h2>
                <p>Unfortunately, your {transaction_type} of <strong>${amount:,.2f}</strong> could not be processed.</p>
                <div style="background: #7f1d1d; padding: 15px; border-radius: 6px; margin: 20px 0;">
                    <p style="margin: 0;"><strong>Reason:</strong> {reason}</p>
                </div>
                <p>What to do next:</p>
                <ul>
                    <li>Check your payment method details</li>
                    <li>Ensure sufficient funds/balance</li>
                    <li>Try a different payment method</li>
                    <li>Contact support if the issue persists</li>
                </ul>
                <div style="margin: 30px 0;">
                    <a href="https://app.ciftmarkets.com/funding" style="background: #00d4ff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">
                        Try Again
                    </a>
                </div>
            </div>
        </body>
        </html>
        """

        await self.send_email(email, subject, html_body)


# Singleton instance
email_service = EmailService()
