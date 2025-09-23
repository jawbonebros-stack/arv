# ARVLab Notification Service
# Handles email notifications, activity tracking, and notification preferences

import json
import requests
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from sqlmodel import Session, select
from main import (
    NotificationSettings, UserNotificationPreferences, 
    NotificationLog, UserActivity, User, Trial, Prediction
)

class NotificationService:
    """Service for managing all types of notifications in ARVLab"""
    
    def __init__(self, db_session: Session):
        self.session = db_session
        self.replit_mail_enabled = self._check_replit_mail_available()
    
    def _check_replit_mail_available(self) -> bool:
        """Check if Replit Mail service is available"""
        return bool(os.getenv("REPL_IDENTITY") or os.getenv("WEB_REPL_RENEWAL"))
    
    def _get_auth_token(self) -> Optional[str]:
        """Get Replit authentication token"""
        if os.getenv("REPL_IDENTITY"):
            return "repl " + os.getenv("REPL_IDENTITY")
        elif os.getenv("WEB_REPL_RENEWAL"):
            return "depl " + os.getenv("WEB_REPL_RENEWAL")
        return None
    
    async def send_email(self, to_email: str, subject: str, text_body: str, html_body: Optional[str] = None) -> bool:
        """Send email using Replit Mail service"""
        if not self.replit_mail_enabled:
            print(f"Email service not available. Would send: {subject}")
            return False
            
        auth_token = self._get_auth_token()
        if not auth_token:
            print("No authentication token found for email service")
            return False
        
        try:
            import httpx
            
            payload = {
                "to": to_email,
                "subject": subject,
                "text": text_body
            }
            if html_body:
                payload["html"] = html_body
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://connectors.replit.com/api/v2/mailer/send",
                    headers={
                        "Content-Type": "application/json",
                        "X-Replit-Token": auth_token.replace("repl ", "").replace("depl ", ""),  # Remove prefixes
                    },
                    json=payload
                )
                
                if 200 <= response.status_code < 300:
                    return True
                else:
                    print(f"Email send failed: {response.status_code}")
                    return False
                    
        except ImportError:
            print("httpx not available, falling back to requests")
            # Fallback to synchronous requests
            try:
                import requests
                payload = {
                    "to": to_email,
                    "subject": subject,
                    "text": text_body
                }
                if html_body:
                    payload["html"] = html_body
                
                response = requests.post(
                    "https://connectors.replit.com/api/v2/mailer/send",
                    headers={
                        "Content-Type": "application/json",
                        "X-Replit-Token": auth_token.replace("repl ", "").replace("depl ", ""),
                    },
                    json=payload,
                    timeout=10
                )
                
                return 200 <= response.status_code < 300
                
            except Exception as e:
                print(f"Email send error: {e}")
                return False
                
        except Exception as e:
            print(f"Email send error: {e}")
            return False
    
    def get_notification_settings(self) -> Dict[str, Any]:
        """Get current admin notification settings"""
        settings = {}
        db_settings = self.session.exec(select(NotificationSettings)).all()
        
        for setting in db_settings:
            try:
                # Try to parse as JSON, fallback to string value
                settings[setting.setting_key] = json.loads(setting.setting_value)
            except (json.JSONDecodeError, TypeError):
                settings[setting.setting_key] = setting.setting_value
        
        # Default settings if not configured
        defaults = {
            "email_notifications_enabled": True,
            "activity_feed_enabled": True,
            "new_results_badges_enabled": True,
            "browser_notifications_enabled": True,
            "email_sender_name": "ARVLab Platform",
            "notification_delay_minutes": 5  # Wait 5 min before sending to batch notifications
        }
        
        for key, value in defaults.items():
            if key not in settings:
                settings[key] = value
                
        return settings
    
    def update_notification_setting(self, setting_key: str, setting_value: Any, updated_by_user_id: int):
        """Update or create a notification setting"""
        # Convert value to JSON string if it's not already a string
        if isinstance(setting_value, (dict, list, bool)):
            value_str = json.dumps(setting_value)
        else:
            value_str = str(setting_value)
        
        existing = self.session.exec(
            select(NotificationSettings).where(NotificationSettings.setting_key == setting_key)
        ).first()
        
        if existing:
            existing.setting_value = value_str
            existing.updated_by = updated_by_user_id
            existing.updated_at = datetime.now(timezone.utc)
        else:
            new_setting = NotificationSettings(
                setting_key=setting_key,
                setting_value=value_str,
                updated_by=updated_by_user_id
            )
            self.session.add(new_setting)
        
        self.session.commit()
    
    def get_user_preferences(self, user_id: int) -> UserNotificationPreferences:
        """Get user's notification preferences, create defaults if not exist"""
        prefs = self.session.exec(
            select(UserNotificationPreferences).where(UserNotificationPreferences.user_id == user_id)
        ).first()
        
        if not prefs:
            # Create default preferences
            prefs = UserNotificationPreferences(user_id=user_id)
            self.session.add(prefs)
            self.session.commit()
            
        return prefs
    
    def update_user_activity(self, user_id: int, activity_type: str):
        """Update user's last activity timestamp"""
        activity = self.session.exec(
            select(UserActivity).where(UserActivity.user_id == user_id)
        ).first()
        
        now = datetime.now(timezone.utc)
        
        if not activity:
            activity = UserActivity(user_id=user_id)
            self.session.add(activity)
        
        if activity_type == "dashboard_visit":
            activity.last_dashboard_visit = now
        elif activity_type == "trial_check":
            activity.last_trial_check = now
        elif activity_type == "notification_check":
            activity.last_notification_check = now
            
        activity.updated_at = now
        self.session.commit()
    
    def get_unseen_concluded_tasks(self, user_id: int) -> List[Dict]:
        """Get tasks user participated in that concluded since their last check"""
        user_activity = self.session.exec(
            select(UserActivity).where(UserActivity.user_id == user_id)
        ).first()
        
        last_check = user_activity.last_notification_check if user_activity else None
        if not last_check:
            # If never checked, look back 30 days
            from datetime import timedelta
            last_check = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Get trials user participated in that were settled after their last check
        user_predictions = self.session.exec(
            select(Prediction).where(Prediction.user_id == user_id)
        ).all()
        
        trial_ids = [p.trial_id for p in user_predictions]
        if not trial_ids:
            return []
        
        concluded_trials = self.session.exec(
            select(Trial)
            .where(Trial.id.in_(trial_ids))
            .where(Trial.status == "settled")
            .where(Trial.result_time_utc > last_check)
        ).all()
        
        results = []
        for trial in concluded_trials:
            # Get user's prediction for this trial
            user_prediction = next((p for p in user_predictions if p.trial_id == trial.id), None)
            
            results.append({
                "trial_id": trial.id,
                "title": trial.title,
                "domain": trial.domain,
                "settled_at": trial.result_time_utc,
                "was_correct": user_prediction.is_correct if user_prediction else None,
                "target_number": trial.target_number
            })
        
        return results
    
    async def send_task_conclusion_email(self, user_id: int, trial: Trial, was_correct: bool):
        """Send email notification when a task the user participated in concludes"""
        settings = self.get_notification_settings()
        if not settings.get("email_notifications_enabled", True):
            return False
            
        user_prefs = self.get_user_preferences(user_id)
        if not user_prefs.email_task_conclusions:
            return False
        
        # Check if we already sent this notification
        existing_log = self.session.exec(
            select(NotificationLog)
            .where(NotificationLog.user_id == user_id)
            .where(NotificationLog.trial_id == trial.id)
            .where(NotificationLog.notification_type == "email_conclusion")
        ).first()
        
        if existing_log:
            return False  # Already sent
        
        # Get user email
        user = self.session.get(User, user_id)
        if not user:
            return False
        
        # Create email content
        result_text = "correct" if was_correct else "incorrect"
        credit_text = "You've earned 1 credit for your accurate prediction!" if was_correct else "Better luck next time!"
        
        subject = f"Task #{trial.target_number} Results Available - ARVLab"
        
        text_body = f"""Hello {user.name},

The ARV task you participated in has been concluded!

Task: {trial.title} (#{trial.target_number})
Domain: {trial.domain.title()}
Your prediction: {result_text}
{credit_text}

View full results and analysis: https://arvlab.xyz/trials/{trial.id}

Continue your ARV journey: https://arvlab.xyz/dashboard

Best regards,
ARVLab Platform
"""
        
        html_body = f"""
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2 style="color: #2563eb;">Task Results Available</h2>
    
    <div style="background: #f8fafc; padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h3 style="margin-top: 0; color: #1e293b;">Task #{trial.target_number}</h3>
        <p><strong>Title:</strong> {trial.title}</p>
        <p><strong>Domain:</strong> {trial.domain.title()}</p>
        <p><strong>Your prediction:</strong> <span style="color: {'#16a34a' if was_correct else '#dc2626'}; font-weight: bold;">{result_text.title()}</span></p>
        {f'<p style="color: #16a34a; font-weight: bold;">🎉 {credit_text}</p>' if was_correct else f'<p style="color: #6b7280;">{credit_text}</p>'}
    </div>
    
    <div style="text-align: center; margin: 30px 0;">
        <a href="https://arvlab.xyz/trials/{trial.id}" style="background: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold;">View Full Results</a>
    </div>
    
    <div style="text-align: center; margin: 20px 0;">
        <a href="https://arvlab.xyz/dashboard" style="color: #2563eb; text-decoration: none;">Continue Your ARV Journey →</a>
    </div>
    
    <hr style="margin: 30px 0; border: none; border-top: 1px solid #e2e8f0;">
    <p style="color: #6b7280; font-size: 14px;">
        You received this email because you participated in ARV task #{trial.target_number}. 
        You can manage your notification preferences in your <a href="https://arvlab.xyz/dashboard">dashboard</a>.
    </p>
</div>
"""
        
        # Send email
        success = await self.send_email(user.email, subject, text_body, html_body)
        
        # Log the notification attempt
        log_entry = NotificationLog(
            user_id=user_id,
            trial_id=trial.id,
            notification_type="email_conclusion",
            status="sent" if success else "failed",
            details=json.dumps({"was_correct": was_correct})
        )
        self.session.add(log_entry)
        self.session.commit()
        
        return success
    
    def create_activity_feed_data(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Create activity feed showing recent task conclusions for user"""
        settings = self.get_notification_settings()
        if not settings.get("activity_feed_enabled", True):
            return []
            
        user_prefs = self.get_user_preferences(user_id)
        if not user_prefs.activity_feed_enabled:
            return []
        
        # Get recent concluded tasks user participated in
        user_predictions = self.session.exec(
            select(Prediction).where(Prediction.user_id == user_id)
        ).all()
        
        trial_ids = [p.trial_id for p in user_predictions]
        if not trial_ids:
            return []
        
        from datetime import timedelta
        thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
        
        concluded_trials = self.session.exec(
            select(Trial)
            .where(Trial.id.in_(trial_ids))
            .where(Trial.status == "settled")
            .where(Trial.result_time_utc > thirty_days_ago)
            .order_by(Trial.result_time_utc.desc())
            .limit(limit)
        ).all()
        
        activity_items = []
        for trial in concluded_trials:
            user_prediction = next((p for p in user_predictions if p.trial_id == trial.id), None)
            
            activity_items.append({
                "type": "task_concluded",
                "trial_id": trial.id,
                "title": trial.title,
                "domain": trial.domain,
                "target_number": trial.target_number,
                "concluded_at": trial.result_time_utc,
                "was_correct": user_prediction.is_correct if user_prediction else None,
                "time_ago": self._get_time_ago(trial.result_time_utc)
            })
        
        return activity_items
    
    def create_activity_feed_data(self, user_id: int, limit: int = 10) -> Dict[str, Any]:
        """Create activity feed data for dashboard display"""
        activity_items = self.get_recent_activity_for_user(user_id)
        unseen_tasks = self.get_unseen_concluded_tasks(user_id)
        
        return {
            "activity_feed": activity_items[:limit],  # Limit items as requested
            "unseen_count": len(unseen_tasks)
        }
    
    def _get_time_ago(self, timestamp: datetime) -> str:
        """Get human-readable time ago string"""
        now = datetime.now(timezone.utc)
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "just now"

# Helper functions for use in main.py routes
async def notify_task_conclusion(db_session: Session, trial_id: int):
    """Send notifications to all participants when a task concludes"""
    service = NotificationService(db_session)
    settings = service.get_notification_settings()
    
    if not settings.get("email_notifications_enabled", True):
        return
    
    # Get all users who participated in this trial
    trial = db_session.get(Trial, trial_id)
    if not trial or trial.status != "settled":
        return
    
    participants = db_session.exec(
        select(Prediction).where(Prediction.trial_id == trial_id)
    ).all()
    
    # Send notifications to each participant
    for prediction in participants:
        try:
            await service.send_task_conclusion_email(
                user_id=prediction.user_id,
                trial=trial,
                was_correct=prediction.is_correct or False
            )
        except Exception as e:
            print(f"Failed to send notification to user {prediction.user_id}: {e}")

def get_notification_service(db_session: Session) -> NotificationService:
    """Factory function to create notification service instance"""
    return NotificationService(db_session)

def schedule_task_conclusion_notification(db_session: Session, trial_id: int):
    """Sync wrapper to schedule async notification in background"""
    import asyncio
    import threading
    
    def run_notification():
        # Create new event loop for the thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(notify_task_conclusion(db_session, trial_id))
        finally:
            loop.close()
    
    # Run in background thread to avoid blocking
    thread = threading.Thread(target=run_notification)
    thread.daemon = True
    thread.start()