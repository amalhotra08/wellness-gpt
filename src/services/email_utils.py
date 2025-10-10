# src/services/email_utils.py
import os
import smtplib
from email.message import EmailMessage

def send_email(to_email: str, subject: str, body: str) -> dict:
    """
    Simple SMTP sender. Configure via env:
      SMTP_HOST, SMTP_PORT, SMTP_STARTTLS (1/0), SMTP_USER, SMTP_PASS, FROM_EMAIL
    For Gmail: use an App Password (not your raw password).
    """
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    use_starttls = os.getenv("SMTP_STARTTLS", "1") == "1"
    user = os.getenv("SMTP_USER", "")
    pwd  = os.getenv("SMTP_PASS", "")
    from_addr = os.getenv("FROM_EMAIL", user or "no-reply@example.com")

    if not to_email:
        return {"ok": False, "error": "Missing recipient email"}

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_email
    msg.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=20) as s:
            if use_starttls:
                s.starttls()
            if user:
                s.login(user, pwd)
            s.send_message(msg)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
