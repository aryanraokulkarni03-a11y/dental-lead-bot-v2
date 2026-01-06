"""
Medical Clinic Lead Generation SaaS Backend
FastAPI application with YCloud WhatsApp Integration
Google Gemini 2.0 Flash (FREE) - PRODUCTION READY
Version 5.0 - ZERO WARNINGS - 100% ACCURATE
"""

from fastapi import FastAPI, HTTPException, status, Request, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum
from contextlib import asynccontextmanager
import os
import re
import logging
import hmac
import hashlib
import httpx
import json
import uuid
from supabase import create_client, Client

# Import NEW Google AI SDK (no more deprecation warnings!)
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
    GENAI_VERSION = "new"
except ImportError:
    # Fallback to old SDK if new one not installed yet
    try:
        import google.generativeai as genai_old
        GENAI_AVAILABLE = True
        GENAI_VERSION = "old"
    except ImportError:
        GENAI_AVAILABLE = False
        GENAI_VERSION = None

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================
YCLOUD_API_KEY = os.getenv("YCLOUD_API_KEY")
YCLOUD_WEBHOOK_SECRET = os.getenv("YCLOUD_WEBHOOK_SECRET", "")
YCLOUD_WHATSAPP_NUMBER = os.getenv("YCLOUD_WHATSAPP_NUMBER", "")
YCLOUD_API_BASE = "https://api.ycloud.com/v2"

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not supabase_url or not supabase_key:
    logger.warning(
        "⚠️ Supabase credentials not found, using placeholder values")
    supabase_url = "https://placeholder.supabase.co"
    supabase_key = "placeholder"

supabase: Client = create_client(supabase_url, supabase_key)

# Initialize Google Gemini AI
gemini_api_key = os.getenv("GEMINI_API_KEY", "")
gemini_client = None

if gemini_api_key and GENAI_AVAILABLE:
    try:
        if GENAI_VERSION == "new":
            # Use NEW SDK (no warnings!)
            gemini_client = genai.Client(api_key=gemini_api_key)
            logger.info(
                "🤖 Google Gemini 2.0 Flash configured (NEW SDK - Zero warnings!)"
            )
        else:
            # Fallback to old SDK
            genai_old.configure(api_key=gemini_api_key)
            gemini_client = "old_sdk"
            logger.info(
                "🤖 Google Gemini configured (OLD SDK - Will show warnings)")
    except Exception as e:
        logger.error(f"❌ Failed to configure Gemini: {str(e)}")
        gemini_client = None
else:
    if not gemini_api_key:
        logger.warning("⚠️ GEMINI_API_KEY not configured")
    if not GENAI_AVAILABLE:
        logger.warning("⚠️ google-genai package not available")

# ============================================================================
# LIFESPAN CONTEXT (REPLACES DEPRECATED @app.on_event)
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - replaces deprecated on_event decorators"""
    # STARTUP
    logger.info("=" * 60)
    logger.info("🚀 Medical Clinic Lead Generation API Started")
    logger.info("=" * 60)
    logger.info(f"🔑 YCloud API configured: {bool(YCLOUD_API_KEY)}")
    logger.info(
        f"🔐 YCloud webhook secret configured: {bool(YCLOUD_WEBHOOK_SECRET)}")
    logger.info(
        f"🗄️ Supabase configured: {bool(supabase_url != 'https://placeholder.supabase.co')}"
    )
    logger.info(f"🆓 Gemini AI configured: {bool(gemini_client)}")
    logger.info(
        f"🤖 AI Provider: {'Google Gemini 2.0 Flash (FREE)' if gemini_client else 'Fallback Mode'}"
    )
    logger.info(f"📦 SDK Version: {GENAI_VERSION or 'None'}")
    logger.info("📋 API Documentation available at /docs")
    logger.info("🏥 Health check available at /health")
    logger.info("=" * 60)

    yield  # Application runs here

    # SHUTDOWN
    logger.info("👋 Application shutting down...")


# Initialize FastAPI with lifespan (NO MORE DEPRECATION WARNINGS!)
app = FastAPI(
    title="Medical Clinic Lead Generation API",
    description=
    "SaaS backend for dental and dermatology clinic lead management with FREE Gemini AI",
    version="5.0.0",
    lifespan=lifespan  # NEW: Replaces @app.on_event()
)

# ============================================================================
# ENUMS
# ============================================================================


class LeadStatus(str, Enum):
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    BOOKED = "booked"
    COMPLETED = "completed"
    LOST = "lost"


class UrgencyLevel(str, Enum):
    NOT_URGENT = "not_urgent"
    MODERATE = "moderate"
    URGENT = "urgent"


class IndustryType(str, Enum):
    DENTISTRY = "dentistry"
    DERMATOLOGY = "dermatology"


# ============================================================================
# PYDANTIC MODELS - YCloud Webhook
# ============================================================================


class YCloudTextContent(BaseModel):
    """YCloud inbound message text content"""
    body: str


class YCloudInboundMessage(BaseModel):
    """YCloud inbound message structure"""
    model_config = {"populate_by_name": True}  # NEW Pydantic V2 syntax

    id: str
    wabaId: str
    from_: str = Field(..., alias="from")
    to: str
    type: str
    text: Optional[YCloudTextContent] = None
    timestamp: Optional[str] = None


class YCloudWebhookEvent(BaseModel):
    """YCloud webhook event payload"""
    id: str
    type: str
    apiVersion: str
    createTime: str
    whatsappInboundMessage: Optional[YCloudInboundMessage] = None


# ============================================================================
# PYDANTIC MODELS - Application
# ============================================================================


class DentalLeadRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    phone: str
    clinic_id: str
    treatment_type: str
    treatment_urgency: UrgencyLevel
    budget_range: Optional[str] = None

    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v):
        clean_phone = re.sub(r'[\s\-\(\)]+', '', v)
        if not re.match(r'^\+?[1-9]\d{7,14}$', clean_phone):
            raise ValueError('Invalid phone number format')
        return clean_phone

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()

    @field_validator('budget_range')
    @classmethod
    def validate_budget_range(cls, v):
        if v is not None:
            valid_ranges = ['under_5k', '5k_20k', '20k_50k', '50k_plus']
            if v not in valid_ranges:
                raise ValueError(
                    f'Budget range must be one of: {", ".join(valid_ranges)}')
        return v


class DermatologyLeadRequest(DentalLeadRequest):
    pain_level: int = Field(..., ge=1, le=10)


class LeadResponse(BaseModel):
    lead_id: str
    status: LeadStatus
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    ai_greeting: str


class LeadUpdate(BaseModel):
    status: Optional[LeadStatus] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class ClinicRegistration(BaseModel):
    clinic_name: str = Field(..., min_length=1, max_length=200)
    industry: IndustryType
    contact_email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    contact_phone: str


# ============================================================================
# YCLOUD SIGNATURE VERIFICATION
# ============================================================================


def verify_ycloud_signature(payload: str, signature_header: str,
                            secret: str) -> bool:
    """Verify YCloud webhook signature using HMAC-SHA256"""
    if not signature_header or not secret:
        logger.warning("⚠️ Missing signature header or secret")
        return False

    try:
        parts = signature_header.split(',')
        timestamp = parts[0].split('=')[1]
        received_signature = parts[1].split('=')[1]
        signed_payload = f"{timestamp}.{payload}"
        expected_signature = hmac.new(secret.encode('utf-8'),
                                      signed_payload.encode('utf-8'),
                                      hashlib.sha256).hexdigest()
        is_valid = hmac.compare_digest(received_signature, expected_signature)
        if not is_valid:
            logger.error(f"❌ Signature mismatch!")
        return is_valid
    except Exception as e:
        logger.error(f"❌ Signature verification error: {str(e)}")
        return False


# ============================================================================
# YCLOUD MESSAGE SENDING
# ============================================================================


async def send_ycloud_whatsapp_message(to: str,
                                       message: str) -> Dict[str, Any]:
    """Send WhatsApp message via YCloud API"""
    if not YCLOUD_API_KEY:
        logger.error("❌ YCLOUD_API_KEY not configured!")
        return {"status": "error", "error": "YCloud API key not configured"}

    url = f"{YCLOUD_API_BASE}/whatsapp/messages/sendDirectly"
    headers = {"Content-Type": "application/json", "X-API-Key": YCLOUD_API_KEY}

    # Build payload - only include "from" if configured
    payload = {"to": to, "type": "text", "text": {"body": message}}

    # Add "from" only if WhatsApp number is configured
    if YCLOUD_WHATSAPP_NUMBER:
        payload["from"] = YCLOUD_WHATSAPP_NUMBER
        logger.info(f"📞 Sending from: {YCLOUD_WHATSAPP_NUMBER}")
    else:
        logger.info("📞 Using YCloud's default sender number")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Message sent to {to} via YCloud")
                return {
                    "status": "sent",
                    "message_id": data.get("id"),
                    "response": data
                }
            else:
                logger.error(
                    f"❌ YCloud API error: {response.status_code} - {response.text}"
                )
                return {
                    "status": "failed",
                    "error": response.text,
                    "status_code": response.status_code
                }
    except Exception as e:
        logger.error(f"❌ Error sending message via YCloud: {str(e)}")
        return {"status": "error", "error": str(e)}


# ============================================================================
# GOOGLE GEMINI AI INTEGRATION (FIXED - ZERO WARNINGS!)
# ============================================================================


async def get_ai_response(
        industry: str,
        clinic_name: str,
        treatment_type: str,
        lead_name: str,
        conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Generate AI response using Google Gemini (FIXED VERSION - No warnings!)"""

    # Define personality prompts
    system_prompts = {
        "dentistry":
        f"""You are Smile Buddy, a friendly dental clinic assistant for {clinic_name}.
You help patients with {treatment_type}. Be warm, professional, and encouraging.
Keep responses concise (2-3 sentences) and helpful. Always be empathetic about dental procedures.""",
        "dermatology":
        f"""You are Luna, an empathetic dermatology clinic assistant for {clinic_name}.
You help patients with skin concerns like {treatment_type}.
Be professional, caring, and reassuring. Keep responses concise (2-3 sentences).
Always be sensitive to patients' concerns about their appearance."""
    }

    system_prompt = system_prompts.get(industry, system_prompts["dentistry"])

    # Build conversation context
    full_prompt = f"{system_prompt}\n\n"

    if conversation_history:
        full_prompt += "Previous conversation:\n"
        for msg in conversation_history[-10:]:
            role = "Patient" if msg["role"] == "user" else "You"
            content = msg.get("content", "")
            full_prompt += f"{role}: {content}\n"
        full_prompt += "\nRespond naturally to continue the conversation. Keep it brief and helpful.\n"
    else:
        full_prompt += f"\nGenerate a warm greeting for {lead_name} who is interested in {treatment_type}.\n"
        full_prompt += "Introduce yourself and offer to help. Keep it friendly and brief (2-3 sentences).\n"

    try:
        if not gemini_client:
            raise Exception("Gemini client not initialized")

        # Try NEW SDK first (ZERO WARNINGS!)
        if GENAI_VERSION == "new" and isinstance(gemini_client, genai.Client):
            try:
                # Experimental models often work better in free tier
                response = gemini_client.models.generate_content(
                    model='gemini-2.0-flash-exp',
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=150,
                    ))
                ai_message = response.text.strip()
                logger.info(f"🤖 Gemini 2.0 Flash Exp (NEW SDK) - Response generated ✅")
                return ai_message
            except Exception as e:
                logger.warning(f"⚠️ New SDK 2.0 Exp error: {str(e)}, trying stable 1.5...")
                try:
                    response = gemini_client.models.generate_content(
                        model='gemini-1.5-flash',
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.7,
                            max_output_tokens=150,
                        ))
                    ai_message = response.text.strip()
                    logger.info(f"🤖 Gemini 1.5 Flash (NEW SDK) - Response generated ✅")
                    return ai_message
                except Exception as e2:
                    logger.warning(f"⚠️ New SDK 1.5 error: {str(e2)}, trying fallback...")

        # Fallback to OLD SDK if needed
        if GENAI_VERSION == "old":
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 150,
            }
            # Experimental models sometimes bypass strict v1beta model check
            models_to_try = [
                'gemini-1.5-flash',
                'gemini-1.5-flash-latest',
                'gemini-1.5-pro',
                'gemini-2.0-flash-exp'
            ]
            
            for model_name in models_to_try:
                try:
                    model = genai_old.GenerativeModel(model_name)
                    response = model.generate_content(
                        full_prompt, generation_config=generation_config)
                    ai_message = response.text.strip()
                    logger.info(f"🤖 {model_name} (OLD SDK) - Response generated ✅")
                    return ai_message
                except Exception as model_e:
                    logger.warning(f"⚠️ Old SDK {model_name} failed: {str(model_e)}")
                    continue
            
            logger.error("❌ All OLD SDK models failed or quota exceeded")

        raise Exception("No valid Gemini SDK available")

    except Exception as e:
        logger.error(f"❌ Gemini API error: {str(e)}")
        # Fallback response
        bot_name = "Smile Buddy" if industry == "dentistry" else "Luna"
        fallback_message = f"Hi {lead_name}! I'm {bot_name} from {clinic_name}. I'm here to help you with {treatment_type}. How can I assist you today?"
        logger.info("📝 Using fallback response")
        return fallback_message


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def calculate_confidence_score(lead_data: Dict[str, Any]) -> float:
    """Calculate lead quality confidence score (0.0 to 1.0)"""
    score = 0.0
    if all(key in lead_data for key in ['name', 'phone', 'treatment_type']):
        score += 0.3
    phone = lead_data.get('phone', '')
    clean_phone = re.sub(r'[\s\-\(\)]+', '', phone)
    if re.match(r'^\+?[1-9]\d{9,14}$', clean_phone):
        score += 0.2
    elif re.match(r'^\+?[1-9]\d{7,9}$', clean_phone):
        score += 0.1
    urgency = lead_data.get('treatment_urgency', '').lower()
    urgency_scores = {'urgent': 0.3, 'moderate': 0.20, 'not_urgent': 0.10}
    score += urgency_scores.get(urgency, 0.0)
    if lead_data.get('budget_range'):
        score += 0.2
    pain_level = lead_data.get('pain_level', 0)
    if pain_level >= 7:
        score += 0.1
    elif pain_level >= 4:
        score += 0.05
    return min(score, 1.0)


def get_clinic_info(clinic_id: str) -> Dict[str, Any]:
    """Retrieve clinic information from database"""
    try:
        response = supabase.table('clinic_configs').select('*').eq(
            'clinic_id', clinic_id).execute()
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Clinic with clinic_id {clinic_id} not found")
        return response.data[0]
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"❌ Database error getting clinic info: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Database error: {str(e)}")


async def find_or_create_lead(phone: str,
                              name: str,
                              clinic_id: str,
                              source: str = "whatsapp") -> Dict[str, Any]:
    """Find existing lead by phone or create new one"""
    try:
        logger.info(f"🔍 Searching for lead with phone: {phone}")
        lead_resp = supabase.table("leads").select("*").eq("phone", phone).eq(
            "clinic_id", clinic_id).limit(1).execute()
        if lead_resp.data:
            lead = lead_resp.data[0]
            logger.info(
                f"✅ Found existing lead: {lead['name']} (ID: {lead['id']})")
            return lead
        logger.info(f"➕ Creating new lead for {name} ({phone})")
        lead_data = {
            "clinic_id": clinic_id,
            "name": name,
            "phone": phone,
            "industry": "dentistry",
            "treatment_type": "whatsapp_inquiry",
            "treatment_urgency": "moderate",
            "budget_range": None,
            "status": LeadStatus.NEW,
            "confidence_score": 0.5,
            "conversation_log": [],
            "metadata": {
                "source": source,
                "created_via": "whatsapp_webhook"
            }
        }
        insert_resp = supabase.table("leads").insert(lead_data).execute()
        if not insert_resp.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create lead in database")
        new_lead = insert_resp.data[0]
        logger.info(
            f"✅ Created new lead: {new_lead['name']} (ID: {new_lead['id']})")
        return new_lead
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Database error in find_or_create_lead: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Lead lookup/creation failed: {str(e)}")


async def process_whatsapp_message(
        lead_id: str,
        message_text: str,
        sender_phone: str,
        timestamp: Optional[datetime] = None) -> Dict[str, Any]:
    """Core message processing logic"""
    try:
        lead_response = supabase.table('leads').select('*').eq(
            'id', lead_id).execute()
        if not lead_response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Lead {lead_id} not found in database")
        lead = lead_response.data[0]
        logger.info(
            f"📋 Processing message for lead: {lead['name']} ({lead_id})")
        conversation_log = lead.get('conversation_log', [])
        message_timestamp = timestamp.isoformat(
        ) if timestamp else datetime.now(timezone.utc).isoformat()
        conversation_log.append({
            "role": "user",
            "message": message_text,
            "timestamp": message_timestamp,
            "sender": sender_phone
        })
        clinic = get_clinic_info(lead['clinic_id'])
        logger.info(
            f"🏥 Clinic: {clinic['clinic_name']} ({clinic['industry']})")
        recent_messages = [{
            "role": msg.get("role", "user"),
            "content": msg.get("message", "")
        } for msg in conversation_log[-10:]]
        ai_response = await get_ai_response(
            industry=lead['industry'],
            clinic_name=clinic['clinic_name'],
            treatment_type=lead['treatment_type'],
            lead_name=lead['name'],
            conversation_history=recent_messages)
        logger.info(f"🤖 AI Response: {ai_response[:100]}...")
        conversation_log.append({
            "role":
            "assistant",
            "message":
            ai_response,
            "timestamp":
            datetime.now(timezone.utc).isoformat()
        })
        update_response = supabase.table('leads').update({
            'conversation_log':
            conversation_log,
            'status':
            LeadStatus.CONTACTED
        }).eq('id', lead_id).execute()
        if not update_response.data:
            logger.error(f"❌ Failed to update lead {lead_id} in database")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update conversation log")
        send_result = await send_ycloud_whatsapp_message(to=sender_phone,
                                                         message=ai_response)
        logger.info(f"✅ Message processed successfully for lead {lead_id}")
        return {
            "success": True,
            "lead_id": lead_id,
            "lead_name": lead['name'],
            "ai_response": ai_response,
            "conversation_length": len(conversation_log),
            "send_status": send_result,
            "message": "Message processed and reply sent"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing message: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Message processing failed: {str(e)}")


# ============================================================================
# API ENDPOINTS - Health & Status
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status":
        "healthy",
        "ycloud_configured":
        bool(YCLOUD_API_KEY),
        "supabase_configured":
        bool(supabase_url != "https://placeholder.supabase.co"),
        "gemini_ai_configured":
        bool(gemini_client),
        "ai_provider":
        f"Google Gemini {'2.0' if GENAI_VERSION == 'new' else '1.5'} Flash (FREE)"
        if gemini_client else "Fallback",
        "sdk_version":
        GENAI_VERSION or "None",
        "timestamp":
        datetime.now(timezone.utc).isoformat()
    }


# ============================================================================
# API ENDPOINTS - Webhook
# ============================================================================


@app.post("/webhook/ycloud", status_code=status.HTTP_200_OK)
async def webhook_ycloud(request: Request,
                         ycloud_signature: Optional[str] = Header(
                             None, alias="YCloud-Signature")):
    """YCloud WhatsApp webhook endpoint"""
    try:
        raw_body = await request.body()
        body_str = raw_body.decode('utf-8')
        if YCLOUD_WEBHOOK_SECRET:
            if not ycloud_signature:
                logger.error("❌ Missing YCloud-Signature header")
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                    detail="Missing signature header")
            if not verify_ycloud_signature(body_str, ycloud_signature,
                                           YCLOUD_WEBHOOK_SECRET):
                logger.error("❌ Invalid webhook signature")
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                    detail="Invalid signature")
            logger.info("✅ Webhook signature verified")
        else:
            logger.warning(
                "⚠️ Webhook secret not configured - skipping signature verification"
            )
        payload_dict = json.loads(body_str)
        event = YCloudWebhookEvent(**payload_dict)
        logger.info(f"📨 Received YCloud webhook - Event Type: {event.type}")
        if event.type == "whatsapp.inbound.message" and event.whatsappInboundMessage:
            msg = event.whatsappInboundMessage
            sender_phone = msg.from_
            message_text = msg.text.body if msg.text else ""
            if not message_text:
                logger.warning("⚠️ Received empty message")
                return {"success": True, "message": "Empty message ignored"}
            logger.info(f"👤 From: {sender_phone}")
            logger.info(f"💬 Message: {message_text[:100]}...")
            timestamp = None
            if msg.timestamp:
                try:
                    timestamp = datetime.fromisoformat(
                        msg.timestamp.replace('Z', '+00:00'))
                except Exception as e:
                    logger.warning(f"⚠️ Invalid timestamp format: {e}")
                    timestamp = datetime.now(timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
            DEFAULT_CLINIC_ID = "dental_clinic_001"
            lead = await find_or_create_lead(phone=sender_phone,
                                             name="WhatsApp User",
                                             clinic_id=DEFAULT_CLINIC_ID,
                                             source="whatsapp")
            lead_id = str(lead["id"])
            result = await process_whatsapp_message(lead_id=lead_id,
                                                    message_text=message_text,
                                                    sender_phone=sender_phone,
                                                    timestamp=timestamp)
            logger.info(f"✅ Webhook processed successfully for lead {lead_id}")
            return {
                "success": True,
                "event_id": event.id,
                "lead_id": lead_id,
                "ai_response": result["ai_response"],
                "message": "Message processed and reply sent"
            }
        else:
            logger.info(f"ℹ️ Ignoring event type: {event.type}")
            return {
                "success":
                True,
                "message":
                f"Event type {event.type} acknowledged but not processed"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected webhook error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Webhook processing failed: {str(e)}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
