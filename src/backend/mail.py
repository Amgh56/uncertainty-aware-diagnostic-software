"""Email sending via Mailtrap SMTP using fastapi-mail."""

import os

from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType

conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME", ""),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD", ""),
    MAIL_FROM=os.getenv("MAIL_FROM", "noreply@safedx.com"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", "587")),
    MAIL_SERVER=os.getenv("MAIL_SERVER", "sandbox.smtp.mailtrap.io"),
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
)


async def send_reset_email(to: str, reset_link: str) -> None:
    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                max-width: 520px; margin: 0 auto; padding: 40px 24px; color: #0f172a;
                text-align: center;">
      <div style="margin-bottom: 32px; display: flex; align-items: center;
                  justify-content: center; gap: 8px;">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none"
             xmlns="http://www.w3.org/2000/svg">
          <path d="M22 12H18L15 21L9 3L6 12H2"
                stroke="#2563eb" stroke-width="2.5"
                stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span style="font-size: 18px; font-weight: 700;">SafeDx</span>
      </div>

      <h1 style="font-size: 22px; font-weight: 700; margin: 0 0 8px;">
        Reset your password
      </h1>
      <p style="font-size: 14px; color: #64748b; line-height: 1.7; margin: 0 auto 28px; max-width: 420px;">
        We received a request to reset the password for your SafeDx account.
        Click the button below to choose a new password.
        This link expires in <strong>30 minutes</strong>.
      </p>

      <a href="{reset_link}"
         style="display: inline-block; background: #2563eb; color: #fff;
                font-size: 14px; font-weight: 600; text-decoration: none;
                padding: 12px 36px; border-radius: 10px;">
        Reset Password
      </a>

      <p style="font-size: 12px; color: #94a3b8; margin-top: 32px; line-height: 1.6;">
        If you did not request a password reset, you can safely ignore this email.
        Your password will not be changed.<br/><br/>
        — The SafeDx Team
      </p>
    </div>
    """

    message = MessageSchema(
        subject="Reset your SafeDx password",
        recipients=[to],
        body=html,
        subtype=MessageType.html,
    )
    fm = FastMail(conf)
    await fm.send_message(message)
