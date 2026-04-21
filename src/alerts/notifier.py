import os
import sys
import smtplib
import threading
from email.message import EmailMessage
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger

# Securely load environment keys
load_dotenv()

class AlertManager:
    def __init__(self):
        """Notification Dispatch Engine natively isolated from CV logic."""
        self.email_sender = os.getenv("EMAIL_SENDER")
        self.email_password = os.getenv("EMAIL_PASSWORD")
        self.email_receiver = os.getenv("EMAIL_RECEIVER")
        
        if not self.email_sender or not self.email_password or not self.email_receiver:
            logger.warning("SMTP Config missing in .env! Active push notifications are DISABLED.")
        else:
            logger.info("AlertManager securely mounted SMTP credentials.")

    def notify(self, timestamp, event_type, object_id, details):
        """
        Asynchronously dispatch alert messages parallel to the CV loop to ensure we never drop FPS.
        """
        # 1. Fire Main Channel: Email Alert
        if self.email_sender and self.email_password and self.email_receiver:
            threading.Thread(
                target=self._send_email,
                args=(timestamp, event_type, object_id, details),
                daemon=True # Ensures threads die if system process exits abruptly
            ).start()
            
        # 2. WhatsApp Extensions (Placeholder layout)
        if os.getenv("WHATSAPP_API_KEY"):
            # Future HTTP Post logic lives here securely decoupled 
            pass
        
    def _send_email(self, timestamp, event_type, object_id, details):
        """Wrapped secure SMTP runtime logic."""
        msg = EmailMessage()
        msg['Subject'] = f"🚨 CRITICAL SYSTEM ALERT: {event_type}"
        msg['From'] = self.email_sender
        msg['To'] = self.email_receiver
        
        # Exact Message Formatting Requirements met securely
        body = (
            f"=== SMART INVENTORY SURVEILLANCE REPORT ===\n\n"
            f"[{timestamp} | {event_type} | ID: {object_id} | DETAILS: {details}]\n\n"
            f"Review live security dashboard immediately."
        )
        msg.set_content(body)
        
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(self.email_sender, self.email_password)
                smtp.send_message(msg)
            logger.info(f"Successfully dispatched Network Alert for Tracking ID {object_id}")
        except Exception as e:
            logger.error(f"SMTP Transmission Failure: {e}")
