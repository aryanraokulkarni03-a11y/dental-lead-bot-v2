"""
Medical Clinic Lead Generation SaaS Backend
FastAPI application with YCloud WhatsApp Integration
Using Google Gemini for FREE AI responses
"""

from fastapi import FastAPI, HTTPException, status, Request, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum
import os
import re
import logging
import hmac
import hashlib
import httpx
import json
import uuid
from supabase import create_client, Client
import google.generativeai as genai

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
YCLOUD_API_BASE = "https://api.ycloud.com/v2"

# Initialize FastAPI
app = FastAPI(
    title="Medical Clinic Lead Generation API",
    description=
    "SaaS backend for dental and dermatology clinic lead management with FREE Gemini AI",
    version="2.0.0")

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
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    logger.info("🤖 Google Gemini configured successfully")
else:
    logger.warning(
        "⚠️ GEMINI_API_KEY not configured - AI responses will use fallback")

# Log configuration status
logger.info(f"🔑 YCloud API Key configured: {bool(YCLOUD_API_KEY)}")
logger.info(
    f"🔐 YCloud Webhook Secret configured: {bool(YCLOUD_WEBHOOK_SECRET)}")
logger.info(f"🗄️ Supabase configured: {bool(supabase_url and supabase_key)}")
logger.info(f"🆓 Gemini AI configured: {bool(gemini_api_key)}")

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
    id: str
    wabaId: str
    from_: str = Field(..., alias="from")
    to: str
    type: str
    text: Optional[YCloudTextContent] = None
    timestamp: Optional[str] = None

    class Config:
        populate_by_name = True


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
    """
    Verify YCloud webhook signature using HMAC-SHA256

    Args:
        payload: Raw request body as string
        signature_header: YCloud-Signature header value (format: t={timestamp},s={signature})
        secret: Webhook secret from YCloud

    Returns:
        True if signature is valid, False otherwise
    """
    if not signature_header or not secret:
        logger.warning("⚠️ Missing signature header or secret")
        return False

    try:
        # Parse signature header: "t=1234567890,s=abc123..."
        parts = signature_header.split(',')
        timestamp = parts[0].split('=')[1]
        received_signature = parts[1].split('=')[1]

        # Construct signed payload: {timestamp}.{request_body}
        signed_payload = f"{timestamp}.{payload}"

        # Compute expected signature using HMAC-SHA256
        expected_signature = hmac.new(secret.encode('utf-8'),
                                      signed_payload.encode('utf-8'),
                                      hashlib.sha256).hexdigest()

        # Constant-time comparison
        is_valid = hmac.compare_digest(received_signature, expected_signature)

        if not is_valid:
            logger.error(f"❌ Signature mismatch!")
            logger.error(
                f"Expected: {expected_signature}, Got: {received_signature}")

        return is_valid

    except Exception as e:
        logger.error(f"❌ Signature verification error: {str(e)}")
        return False


# ============================================================================
# YCLOUD MESSAGE SENDING
# ============================================================================


async def send_ycloud_whatsapp_message(to: str,
                                       message: str) -> Dict[str, Any]:
    """
    Send WhatsApp message via YCloud API

    Args:
        to: Recipient phone number (with country code, e.g., +919876543210)
        message: Text message to send

    Returns:
        Dictionary with send status and response data
    """
    if not YCLOUD_API_KEY:
        logger.error("❌ YCLOUD_API_KEY not configured!")
        return {"status": "error", "error": "YCloud API key not configured"}

    url = f"{YCLOUD_API_BASE}/whatsapp/messages/sendDirectly"

    headers = {"Content-Type": "application/json", "X-API-Key": YCLOUD_API_KEY}

    payload = {"to": to, "type": "text", "text": {"body": message}}

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
# GOOGLE GEMINI AI INTEGRATION
# ============================================================================


async def get_ai_response(
        industry: str,
        clinic_name: str,
        treatment_type: str,
        lead_name: str,
        conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Generate AI response using Google Gemini (FREE!)

    Args:
        industry: "dentistry" or "dermatology"
        clinic_name: Name of the clinic
        treatment_type: Type of treatment being discussed
        lead_name: Patient's name
        conversation_history: Previous messages in the conversation

    Returns:
        AI-generated response string
    """

    # Define personality prompts for each industry
    system_prompts = {
        "dentistry":
        f"""You are Smile Buddy, a friendly dental clinic assistant for {clinic_name}.
You help patients with {treatment_type}. You are warm, professional, and encouraging.
Keep your responses concise (2-3 sentences) and helpful.
Always be empathetic and reassuring about dental procedures.""",
        "dermatology":
        f"""You are Luna, an empathetic dermatology clinic assistant for {clinic_name}.
You help patients with skin concerns like {treatment_type}.
You are professional, caring, and reassuring. Keep responses concise (2-3 sentences).
Always be sensitive to patients' concerns about their appearance."""
    }

    system_prompt = system_prompts.get(industry, system_prompts["dentistry"])

    # Build the full conversation context
    full_prompt = f"{system_prompt}\n\n"

    # Add conversation history if exists
    if conversation_history:
        full_prompt += "Previous conversation:\n"
        for msg in conversation_history[-10:]:  # Last 10 messages only
            role = "Patient" if msg["role"] == "user" else "You"
            content = msg.get("content", "")
            full_prompt += f"{role}: {content}\n"
        full_prompt += "\nRespond naturally to continue the conversation. Keep it brief and helpful.\n"
    else:
        # First message - greeting
        full_prompt += f"\nGenerate a warm greeting for {lead_name} who is interested in {treatment_type}.\n"
        full_prompt += "Introduce yourself and offer to help. Keep it friendly and brief (2-3 sentences).\n"

    try:
        if not gemini_api_key:
            raise Exception("Gemini API key not configured")

        # Use Gemini Pro model (FREE!)
        model = genai.GenerativeModel('gemini-pro')

        # Configure generation parameters
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 150,
        }

        # Generate response
        response = model.generate_content(full_prompt,
                                          generation_config=generation_config)

        # Extract text from response
        ai_message = response.text.strip()

        logger.info(f"🤖 Gemini AI generated response successfully")
        return ai_message

    except Exception as e:
        logger.error(f"❌ Gemini API error: {str(e)}")

        # Fallback response if AI fails
        bot_name = "Smile Buddy" if industry == "dentistry" else "Luna"
        fallback_message = f"Hi {lead_name}! I'm {bot_name} from {clinic_name}. I'm here to help you with {treatment_type}. How can I assist you today?"

        logger.info("📝 Using fallback response due to AI error")
        return fallback_message


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def calculate_confidence_score(lead_data: Dict[str, Any]) -> float:
    """Calculate lead quality confidence score (0.0 to 1.0)"""
    score = 0.0

    # Basic info completeness (30%)
    if all(key in lead_data for key in ['name', 'phone', 'treatment_type']):
        score += 0.3

    # Phone number validity (20%)
    phone = lead_data.get('phone', '')
    clean_phone = re.sub(r'[\s\-\(\)]+', '', phone)
    if re.match(r'^\+?[1-9]\d{9,14}$', clean_phone):
        score += 0.2
    elif re.match(r'^\+?[1-9]\d{7,9}$', clean_phone):
        score += 0.1

    # Treatment urgency (30%)
    urgency = lead_data.get('treatment_urgency', '').lower()
    urgency_scores = {'urgent': 0.3, 'moderate': 0.20, 'not_urgent': 0.10}
    score += urgency_scores.get(urgency, 0.0)

    # Budget information (20%)
    if lead_data.get('budget_range'):
        score += 0.2

    # Pain level (dermatology only, 10%)
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
    """Core message processing logic - handles incoming messages and generates AI responses"""
    try:
        # Fetch lead from database
        lead_response = supabase.table('leads').select('*').eq(
            'id', lead_id).execute()

        if not lead_response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Lead {lead_id} not found in database")

        lead = lead_response.data[0]
        logger.info(
            f"📋 Processing message for lead: {lead['name']} ({lead_id})")

        # Get conversation history
        conversation_log = lead.get('conversation_log', [])
        message_timestamp = timestamp.isoformat(
        ) if timestamp else datetime.now(timezone.utc).isoformat()

        # Append user message
        conversation_log.append({
            "role": "user",
            "message": message_text,
            "timestamp": message_timestamp,
            "sender": sender_phone
        })

        # Get clinic info
        clinic = get_clinic_info(lead['clinic_id'])
        logger.info(
            f"🏥 Clinic: {clinic['clinic_name']} ({clinic['industry']})")

        # Prepare conversation history for AI (last 10 messages)
        recent_messages = [{
            "role": msg.get("role", "user"),
            "content": msg.get("message", "")
        } for msg in conversation_log[-10:]]

        # Generate AI response using Gemini
        ai_response = await get_ai_response(
            industry=lead['industry'],
            clinic_name=clinic['clinic_name'],
            treatment_type=lead['treatment_type'],
            lead_name=lead['name'],
            conversation_history=recent_messages)

        logger.info(f"🤖 AI Response: {ai_response[:100]}...")

        # Append assistant response
        conversation_log.append({
            "role":
            "assistant",
            "message":
            ai_response,
            "timestamp":
            datetime.now(timezone.utc).isoformat()
        })

        # Update lead in database
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

        # Send reply via YCloud
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
        bool(gemini_api_key),
        "ai_provider":
        "Google Gemini (FREE)",
        "timestamp":
        datetime.now(timezone.utc).isoformat()
    }


# ============================================================================
# API ENDPOINTS - Lead Management
# ============================================================================


@app.post("/lead/dental",
          response_model=LeadResponse,
          status_code=status.HTTP_201_CREATED)
async def create_dental_lead(lead_request: DentalLeadRequest):
    """Create a new dental clinic lead"""
    try:
        clinic = get_clinic_info(lead_request.clinic_id)

        if clinic["industry"] != IndustryType.DENTISTRY:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="clinic_id is not a dentistry clinic")

        lead_dict = lead_request.model_dump()
        confidence_score = calculate_confidence_score(lead_dict)

        ai_greeting = await get_ai_response(
            industry="dentistry",
            clinic_name=clinic['clinic_name'],
            treatment_type=lead_request.treatment_type,
            lead_name=lead_request.name)

        lead_data = {
            "clinic_id":
            lead_request.clinic_id,
            "name":
            lead_request.name,
            "phone":
            lead_request.phone,
            "industry":
            "dentistry",
            "treatment_type":
            lead_request.treatment_type,
            "treatment_urgency":
            lead_request.treatment_urgency,
            "budget_range":
            lead_request.budget_range,
            "status":
            LeadStatus.NEW,
            "confidence_score":
            confidence_score,
            "conversation_log": [{
                "role":
                "assistant",
                "message":
                ai_greeting,
                "timestamp":
                datetime.now(timezone.utc).isoformat()
            }],
            "metadata": {}
        }

        insert_resp = supabase.table('leads').insert(lead_data).execute()
        lead_id = insert_resp.data[0]["id"]

        return LeadResponse(lead_id=str(lead_id),
                            status=LeadStatus.NEW,
                            confidence_score=confidence_score,
                            ai_greeting=ai_greeting)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to create dental lead: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to create dental lead: {str(e)}")


@app.post("/lead/dermatology",
          response_model=LeadResponse,
          status_code=status.HTTP_201_CREATED)
async def create_dermatology_lead(lead_request: DermatologyLeadRequest):
    """Create a new dermatology clinic lead"""
    try:
        clinic = get_clinic_info(lead_request.clinic_id)

        if clinic["industry"] != IndustryType.DERMATOLOGY:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="clinic_id is not a dermatology clinic")

        lead_dict = lead_request.model_dump()
        confidence_score = calculate_confidence_score(lead_dict)

        ai_greeting = await get_ai_response(
            industry="dermatology",
            clinic_name=clinic['clinic_name'],
            treatment_type=lead_request.treatment_type,
            lead_name=lead_request.name)

        lead_data = {
            "clinic_id":
            lead_request.clinic_id,
            "name":
            lead_request.name,
            "phone":
            lead_request.phone,
            "industry":
            "dermatology",
            "treatment_type":
            lead_request.treatment_type,
            "treatment_urgency":
            lead_request.treatment_urgency,
            "budget_range":
            lead_request.budget_range,
            "pain_level":
            lead_request.pain_level,
            "status":
            LeadStatus.NEW,
            "confidence_score":
            confidence_score,
            "conversation_log": [{
                "role":
                "assistant",
                "message":
                ai_greeting,
                "timestamp":
                datetime.now(timezone.utc).isoformat()
            }],
            "metadata": {}
        }

        insert_resp = supabase.table('leads').insert(lead_data).execute()
        lead_id = insert_resp.data[0]["id"]

        return LeadResponse(lead_id=str(lead_id),
                            status=LeadStatus.NEW,
                            confidence_score=confidence_score,
                            ai_greeting=ai_greeting)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to create dermatology lead: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create dermatology lead: {str(e)}")


@app.get("/leads/{clinic_id}")
async def get_clinic_leads(clinic_id: str):
    """Retrieve all leads for a specific clinic"""
    try:
        get_clinic_info(clinic_id)
        response = supabase.table('leads').select('*').eq(
            'clinic_id', clinic_id).order('created_at', desc=True).execute()
        return {"leads": response.data, "count": len(response.data)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to retrieve leads: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to retrieve leads: {str(e)}")


@app.patch("/leads/{lead_id}")
async def update_lead(lead_id: str, lead_update: LeadUpdate):
    """Update lead status and/or confidence score"""
    try:
        lead_response = supabase.table('leads').select('*').eq(
            'id', lead_id).execute()

        if not lead_response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Lead with id {lead_id} not found")

        update_data = {}
        if lead_update.status is not None:
            update_data['status'] = lead_update.status
        if lead_update.confidence_score is not None:
            update_data['confidence_score'] = lead_update.confidence_score

        if not update_data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="No fields provided for update")

        updated_response = supabase.table('leads').update(update_data).eq(
            'id', lead_id).execute()

        return {
            "lead": updated_response.data[0],
            "message": "Lead updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update lead: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to update lead: {str(e)}")


# ============================================================================
# API ENDPOINTS - Clinic Management
# ============================================================================


@app.post("/clinic/register", status_code=status.HTTP_201_CREATED)
async def register_clinic(clinic: ClinicRegistration):
    """Register a new clinic in the system"""
    try:
        clinic_id = f"{clinic.industry.value}_{uuid.uuid4().hex[:12]}"
        bot_name = "Smile Buddy" if clinic.industry == IndustryType.DENTISTRY else "Luna"

        clinic_data = {
            "clinic_id": clinic_id,
            "clinic_name": clinic.clinic_name,
            "industry": clinic.industry,
            "owner_name": clinic.clinic_name,
            "phone": clinic.contact_phone,
            "email": clinic.contact_email,
            "whatsapp_number": None,
            "wati_api_token": None,
            "bot_name": bot_name,
            "bot_personality":
            f"Default {bot_name} personality - friendly and professional assistant",
            "auto_book_appointments": False
        }

        supabase.table('clinic_configs').insert(clinic_data).execute()

        return {
            "clinic_id": clinic_id,
            "clinic_name": clinic.clinic_name,
            "bot_name": bot_name,
            "message": "Clinic registered successfully"
        }

    except Exception as e:
        logger.error(f"❌ Failed to register clinic: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to register clinic: {str(e)}")


@app.get("/clinic/{clinic_id}/dashboard")
async def get_clinic_dashboard(clinic_id: str):
    """Get dashboard statistics for a clinic"""
    try:
        get_clinic_info(clinic_id)
        response = supabase.table('leads').select('status').eq(
            'clinic_id', clinic_id).execute()

        total_leads = len(response.data)
        new_leads = sum(1 for lead in response.data
                        if lead['status'] == LeadStatus.NEW)
        contacted_leads = sum(1 for lead in response.data
                              if lead['status'] == LeadStatus.CONTACTED)
        qualified_leads = sum(1 for lead in response.data
                              if lead['status'] == LeadStatus.QUALIFIED)
        booked_leads = sum(1 for lead in response.data
                           if lead['status'] == LeadStatus.BOOKED)
        completed_leads = sum(1 for lead in response.data
                              if lead['status'] == LeadStatus.COMPLETED)
        lost_leads = sum(1 for lead in response.data
                         if lead['status'] == LeadStatus.LOST)

        conversion_rate = ((booked_leads + completed_leads) / total_leads *
                           100) if total_leads > 0 else 0.0

        return {
            "clinic_id": clinic_id,
            "total_leads": total_leads,
            "new": new_leads,
            "contacted": contacted_leads,
            "qualified": qualified_leads,
            "booked": booked_leads,
            "completed": completed_leads,
            "lost": lost_leads,
            "conversion_rate": round(conversion_rate, 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to retrieve dashboard data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dashboard data: {str(e)}")


# ============================================================================
# YCLOUD WEBHOOK ENDPOINT
# ============================================================================


@app.post("/webhook/ycloud", status_code=status.HTTP_200_OK)
async def webhook_ycloud(request: Request,
                         ycloud_signature: Optional[str] = Header(
                             None, alias="YCloud-Signature")):
    """
    YCloud WhatsApp webhook endpoint

    Handles incoming WhatsApp messages from YCloud and processes them with Gemini AI
    Supports signature verification for security
    """
    try:
        # Get raw body for signature verification
        raw_body = await request.body()
        body_str = raw_body.decode('utf-8')

        # Verify signature if secret is configured
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

        # Parse webhook payload
        payload_dict = json.loads(body_str)
        event = YCloudWebhookEvent(**payload_dict)

        logger.info(f"📨 Received YCloud webhook - Event Type: {event.type}")

        # Handle inbound WhatsApp messages
        if event.type == "whatsapp.inbound.message" and event.whatsappInboundMessage:
            msg = event.whatsappInboundMessage

            # Extract message details
            sender_phone = msg.from_
            message_text = msg.text.body if msg.text else ""

            if not message_text:
                logger.warning("⚠️ Received empty message")
                return {"success": True, "message": "Empty message ignored"}

            logger.info(f"👤 From: {sender_phone}")
            logger.info(f"💬 Message: {message_text[:100]}...")

            # Parse timestamp
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

            # Determine clinic (TODO: implement multi-clinic routing)
            DEFAULT_CLINIC_ID = "dental_clinic_001"

            # Find or create lead
            lead = await find_or_create_lead(
                phone=sender_phone,
                name=
                "WhatsApp User",  # Default name, will be updated in conversation
                clinic_id=DEFAULT_CLINIC_ID,
                source="whatsapp")

            lead_id = str(lead["id"])

            # Process message and generate AI response
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
# FRONTEND ENDPOINTS
# ============================================================================


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve dashboard HTML"""
    try:
        with open("src/static/dashboard_html.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.error("❌ Dashboard HTML file not found")
        return HTMLResponse(
            content=
            "<h1>Dashboard not found</h1><p>Please ensure src/static/dashboard_html.html exists</p>",
            status_code=404)


# Mount static files
try:
    app.mount("/static", StaticFiles(directory="src/static"), name="static")
except Exception as e:
    logger.warning(f"⚠️ Could not mount static files directory: {str(e)}")

# ============================================================================
# STARTUP & SHUTDOWN EVENTS
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("=" * 60)
    logger.info("🚀 Medical Clinic Lead Generation API Started")
    logger.info("=" * 60)
    logger.info(f"🔑 YCloud API configured: {bool(YCLOUD_API_KEY)}")
    logger.info(
        f"🔐 YCloud webhook secret configured: {bool(YCLOUD_WEBHOOK_SECRET)}")
    logger.info(
        f"🗄️ Supabase configured: {bool(supabase_url != 'https://placeholder.supabase.co')}"
    )
    logger.info(f"🆓 Gemini AI configured: {bool(gemini_api_key)}")
    logger.info(
        f"🤖 AI Provider: Google Gemini (FREE - 15 req/min, 1M tokens/day)")
    logger.info("📋 API Documentation available at /docs")
    logger.info("🏥 Health check available at /health")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Application shutting down...")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
