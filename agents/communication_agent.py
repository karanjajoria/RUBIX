"""
Crisis Communication Agent
Sends multi-lingual alerts to humanitarian organizations via SMS/email.
Implements sequential workflow: receives alerts from Orchestrator → sends notifications.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

import google.generativeai as genai

try:
    from twilio.rest import Client
except ImportError:
    print("Warning: Twilio not installed. SMS features disabled.")
    Client = None

from config.config import (
    ModelConfig,
    CommunicationConfig,
    GEMINI_API_KEY,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER
)
from utils.memory import memory_manager
from utils.ollama_client import ollama_client


class CrisisCommunicationAgent:
    """
    Agent 4: Crisis Communication
    Sends timely, multi-lingual alerts to humanitarian stakeholders.

    Multi-Agent Feature: SEQUENTIAL workflow endpoint
    - Receives resource calculations from Resource Agent
    - Generates and sends alerts to stakeholders
    """

    def __init__(self):
        """Initialize Communication Agent with Gemini and Twilio."""
        self.agent_name = "crisis_communication"

        # Configure Gemini for translation and alert generation
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel(ModelConfig.GEMINI_FLASH)

        # Initialize Twilio client (if credentials available)
        self.twilio_client = None
        if Client and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
            try:
                self.twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                print(f"[Communication Agent] Twilio initialized")
            except Exception as e:
                print(f"[Communication Agent] Twilio initialization failed: {e}")

        # Configuration
        self.supported_languages = CommunicationConfig.SUPPORTED_LANGUAGES
        self.alert_templates = CommunicationConfig.ALERT_TEMPLATES
        self.priority_levels = CommunicationConfig.PRIORITY_LEVELS

    def send_alert(self, alert_type: str, data: Dict[str, Any],
                   recipients: List[str], languages: Optional[List[str]] = None,
                   priority: str = "medium") -> Dict[str, Any]:
        """
        Send crisis alert to humanitarian organizations.

        Sequential Workflow Integration:
        - Input: Data from Resource Agent (resource needs, locations)
        - Output: Alert delivery confirmations

        Args:
            alert_type: Type of alert ("displacement_warning", "resource_need")
            data: Alert data (displacement count, location, resources, etc.)
            recipients: List of phone numbers or emails
            languages: Languages for multi-lingual alerts (defaults to ["en"])
            priority: Alert priority ("low", "medium", "high", "urgent")

        Returns:
            Dictionary containing:
                - alerts_sent: Number of alerts successfully sent
                - delivery_status: Status per recipient
                - alert_content: Generated alert messages
                - failed_recipients: List of failed deliveries
        """
        start_time = datetime.now()

        try:
            # Use default language if none specified
            if languages is None:
                languages = ["en"]

            # Step 1: Generate alert content in each language using Gemini
            alert_messages = self._generate_multi_lingual_alerts(
                alert_type=alert_type,
                data=data,
                languages=languages,
                priority=priority
            )

            # Step 2: Send alerts to recipients
            delivery_status = []
            alerts_sent = 0
            failed_recipients = []

            for recipient in recipients:
                for lang, message in alert_messages.items():
                    success = self._send_message(
                        recipient=recipient,
                        message=message,
                        priority=priority
                    )

                    if success:
                        alerts_sent += 1
                        delivery_status.append({
                            "recipient": recipient,
                            "language": lang,
                            "status": "delivered",
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        failed_recipients.append(recipient)
                        delivery_status.append({
                            "recipient": recipient,
                            "language": lang,
                            "status": "failed",
                            "timestamp": datetime.now().isoformat()
                        })

            # Prepare result
            execution_time = (datetime.now() - start_time).total_seconds()
            result = {
                "alerts_sent": alerts_sent,
                "delivery_status": delivery_status,
                "alert_content": alert_messages,
                "failed_recipients": list(set(failed_recipients)),
                "priority": priority,
                "alert_type": alert_type,
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time
            }

            # Log to episodic memory
            memory_manager.episodic_memory.log_episode(
                agent_name=self.agent_name,
                action="send_alert",
                input_data={"alert_type": alert_type, "recipients": recipients, "priority": priority},
                output_data=result,
                status="success",
                metadata={"execution_time": execution_time}
            )

            return result

        except Exception as e:
            # Log error
            error_result = {
                "error": str(e),
                "alert_type": alert_type,
                "recipients": recipients
            }

            memory_manager.episodic_memory.log_episode(
                agent_name=self.agent_name,
                action="send_alert",
                input_data={"alert_type": alert_type, "recipients": recipients},
                output_data=error_result,
                status="error",
                metadata={"error_message": str(e)}
            )

            return error_result

    def _generate_multi_lingual_alerts(self, alert_type: str, data: Dict[str, Any],
                                      languages: List[str], priority: str) -> Dict[str, str]:
        """
        Generate alert messages in multiple languages using Gemini.

        Context Engineering: Multi-lingual, culturally-appropriate messaging
        """
        messages = {}

        for lang in languages:
            # Get template if available
            template = self.alert_templates.get(alert_type, {}).get(lang, "")

            if template:
                # Fill template with data
                try:
                    message = template.format(**data)
                    messages[lang] = message
                except KeyError:
                    # Template formatting failed, use Gemini
                    messages[lang] = self._gemini_translate_alert(
                        alert_type=alert_type,
                        data=data,
                        target_language=lang,
                        priority=priority
                    )
            else:
                # No template, use Gemini to generate
                messages[lang] = self._gemini_translate_alert(
                    alert_type=alert_type,
                    data=data,
                    target_language=lang,
                    priority=priority
                )

        return messages

    def _gemini_translate_alert(self, alert_type: str, data: Dict[str, Any],
                               target_language: str, priority: str) -> str:
        """
        Use Gemini to generate culturally-appropriate alert in target language.

        Context Engineering: Language-specific, humanitarian tone
        """
        # Map language codes to full names
        lang_names = {
            "en": "English",
            "fr": "French",
            "ar": "Arabic",
            "sw": "Swahili"
        }
        lang_name = lang_names.get(target_language, "English")

        # Prepare data summary
        if alert_type == "displacement_warning":
            data_summary = f"""
- Predicted Displacement: {data.get('count', 'Unknown')} people
- Location: {data.get('location', 'Unknown')}
- Timeline: {data.get('months', 'Unknown')} months
- Threat Level: {data.get('level', 'Unknown')}
"""
        elif alert_type == "resource_need":
            data_summary = f"""
- Location: {data.get('location', 'Unknown')}
- Affected Population: {data.get('count', 'Unknown')} people
- Required Resources: {data.get('resources', 'Unknown')}
"""
        else:
            data_summary = str(data)

        prompt = f"""You are a humanitarian crisis communication specialist.

Task: Generate a {priority.upper()} priority alert message in {lang_name} for humanitarian organizations.

Alert Type: {alert_type.replace('_', ' ').title()}

Alert Data:
{data_summary}

Requirements:
1. Clear, concise message (SMS-friendly, under 160 characters if possible)
2. Culturally appropriate for {lang_name} speakers
3. Action-oriented language
4. Professional humanitarian tone
5. Include urgency indicator for {priority} priority

Generate the alert message in {lang_name} only (no translation notes or explanations)."""

        try:
            import time
            max_retries = 3
            retry_delay = 2  # seconds

            # Try Gemini first
            for attempt in range(max_retries):
                try:
                    response = self.gemini_model.generate_content(prompt)
                    return response.text.strip()
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        if attempt < max_retries - 1:
                            print(f"[Communication Agent] Gemini quota limit hit, retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            print(f"[Communication Agent] Gemini quota exhausted, switching to Ollama (Llama 3)...")
                            break
                    elif "API_KEY" in str(e) or "ADC" in str(e):
                        print(f"[Communication Agent] Gemini API key not configured, using Ollama (Llama 3)...")
                        break
                    raise e

            # Fallback to Ollama (Llama 3)
            if ollama_client.is_available():
                print(f"[Communication Agent] Using Ollama (Llama 3) for alert generation...")
                ollama_response = ollama_client.generate(
                    prompt=prompt,
                    system="You are a humanitarian communication AI assistant specializing in crisis alerts.",
                    temperature=0.7,
                    max_tokens=200
                )
                if ollama_response:
                    return ollama_response

            # Final fallback if both fail
            return f"ALERT ({priority.upper()}): {alert_type.replace('_', ' ').title()} - {data.get('location', 'Unknown location')}"

        except Exception as e:
            print(f"[Communication Agent] LLM translation error: {e}")
            # Fallback to English template
            return f"ALERT ({priority.upper()}): {alert_type.replace('_', ' ').title()} - {data.get('location', 'Unknown location')}"

    def _send_message(self, recipient: str, message: str, priority: str) -> bool:
        """
        Send message via SMS (Twilio) or email.

        Returns True if sent successfully, False otherwise.
        """
        # Determine if recipient is phone number or email
        is_phone = recipient.startswith("+") or recipient.isdigit()

        if is_phone:
            return self._send_sms(recipient, message, priority)
        else:
            return self._send_email(recipient, message, priority)

    def _send_sms(self, phone_number: str, message: str, priority: str) -> bool:
        """Send SMS via Twilio."""
        if self.twilio_client is None:
            print(f"[Communication Agent] Twilio not configured. Mock SMS to {phone_number}: {message[:50]}...")
            return True  # Mock success for demo

        try:
            # Send SMS
            message_obj = self.twilio_client.messages.create(
                body=message,
                from_=TWILIO_PHONE_NUMBER,
                to=phone_number
            )

            print(f"[Communication Agent] SMS sent to {phone_number}: {message_obj.sid}")
            return True

        except Exception as e:
            print(f"[Communication Agent] SMS send error to {phone_number}: {e}")
            return False

    def _send_email(self, email: str, message: str, priority: str) -> bool:
        """
        Send email notification.

        For demo, mock email sending (in production, use SendGrid, SES, etc.)
        """
        print(f"[Communication Agent] Email sent to {email} (Priority: {priority}):\n{message}")
        return True  # Mock success for demo

    def broadcast_alert(self, alert_type: str, data: Dict[str, Any],
                       organization_type: str = "all") -> Dict[str, Any]:
        """
        Broadcast alert to predefined humanitarian organization groups.

        Args:
            alert_type: Type of alert
            data: Alert data
            organization_type: Target organization type ("un", "ngo", "government", "all")

        Returns:
            Broadcast delivery summary
        """
        # Predefined organization contacts (demo data)
        organization_contacts = {
            "un": ["+1234567890", "unhcr@example.org"],  # UNHCR
            "ngo": ["+0987654321", "msf@example.org"],  # Doctors Without Borders
            "government": ["gov@example.org"],  # Local government
        }

        # Select recipients
        if organization_type == "all":
            recipients = [contact for contacts in organization_contacts.values() for contact in contacts]
        else:
            recipients = organization_contacts.get(organization_type, [])

        # Determine priority from data
        threat_level = data.get("level", "medium")
        priority_map = {"critical": "urgent", "high": "high", "medium": "medium", "low": "low"}
        priority = priority_map.get(threat_level, "medium")

        # Send alert
        return self.send_alert(
            alert_type=alert_type,
            data=data,
            recipients=recipients,
            languages=["en", "fr"],  # Multi-lingual for broader reach
            priority=priority
        )


# Create global instance
communication_agent = CrisisCommunicationAgent()
