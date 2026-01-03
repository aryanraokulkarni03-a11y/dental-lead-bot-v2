"""
Medical Clinic Lead Generation SaaS Backend
FastAPI application for dental and dermatology clinic lead management with AI integration
"""

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum
import os
import re
import logging
from openai import AsyncOpenAI
from supabase import create_client, Client
import uuid

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
print("SUPABASE_URL from env:", os.getenv("SUPABASE_URL"))
print("SERVICE_ROLE set?:", bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY")))

app = FastAPI(title="Medical Clinic Lead Generation API",
              description=
              "SaaS backend for dental and dermatology clinic lead management",
              version="1.0.0")

# Initialize clients
openai_url = os.getenv("SUPABASE_URL")
openai_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not openai_url or not openai_key:
    # Fallback or dummy for initialization if env vars missing during startup
    # In a real app, you'd want to handle this more gracefully
    openai_url = "https://placeholder.supabase.co"
    openai_key = "placeholder"

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy_key"))
supabase: Client = create_client(openai_url, openai_key)

# ============================================================================
# ENUMS - ALIGNED WITH SQL SCHEMA
# ============================================================================


class LeadStatus(str, Enum):
    """Lead status enumeration - matches lead_status SQL enum"""
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    BOOKED = "booked"
    COMPLETED = "completed"
    LOST = "lost"


class UrgencyLevel(str, Enum):
    """Treatment urgency levels - matches urgency_level SQL enum"""
    NOT_URGENT = "not_urgent"
    MODERATE = "moderate"
    URGENT = "urgent"


class IndustryType(str, Enum):
    """Industry types - matches industry_type SQL enum"""
    DENTISTRY = "dentistry"
    DERMATOLOGY = "dermatology"


# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class DentalLeadRequest(BaseModel):
    """Request model for dental clinic leads"""
    name: str = Field(...,
                      min_length=1,
                      max_length=100,
                      description="Patient name")
    phone: str = Field(..., description="Patient phone number")
    clinic_id: str = Field(..., description="Clinic identifier")
    treatment_type: str = Field(...,
                                description="Type of dental treatment needed")
    treatment_urgency: UrgencyLevel = Field(
        ..., description="Urgency level of treatment")
    budget_range: Optional[str] = Field(
        None,
        description=
        "Patient's budget range: under_5k, 5k_20k, 20k_50k, 50k_plus")

    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v):
        """Validate phone number format"""
        # Remove common separators
        clean_phone = re.sub(r'[\s\-\(\)]+', '', v)
        if not re.match(r'^\+?[1-9]\d{7,14}$', clean_phone):
            raise ValueError('Invalid phone number format')
        return clean_phone

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate name is not empty or just whitespace"""
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()

    @field_validator('budget_range')
    @classmethod
    def validate_budget_range(cls, v):
        """Validate budget_range matches SQL enum values"""
        if v is not None:
            valid_ranges = ['under_5k', '5k_20k', '20k_50k', '50k_plus']
            if v not in valid_ranges:
                raise ValueError(
                    f'Budget range must be one of: {", ".join(valid_ranges)}')
        return v


class DermatologyLeadRequest(DentalLeadRequest):
    """Request model for dermatology clinic leads"""
    pain_level: int = Field(...,
                            ge=1,
                            le=10,
                            description="Pain level from 1-10")


class LeadResponse(BaseModel):
    """Response model for lead creation"""
    lead_id: str = Field(..., description="Unique lead identifier")
    status: LeadStatus = Field(..., description="Current lead status")
    confidence_score: float = Field(...,
                                    ge=0.0,
                                    le=1.0,
                                    description="Lead quality score 0-1")
    ai_greeting: str = Field(..., description="AI-generated greeting message")


class LeadUpdate(BaseModel):
    """Request model for updating lead"""
    status: Optional[LeadStatus] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class ClinicRegistration(BaseModel):
    """Request model for clinic registration"""
    clinic_name: str = Field(..., min_length=1, max_length=200)
    industry: IndustryType = Field(
        ..., description="Industry type: dentistry or dermatology")
    contact_email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    contact_phone: str


class WhatsAppWebhook(BaseModel):
    """Request model for WhatsApp webhook"""
    lead_id: str
    message: str
    timestamp: Optional[datetime] = None
    sender: str


class WATITextMessage(BaseModel):
    """WATI text message structure"""
    body: Optional[str] = ""


class WATIMessage(BaseModel):
    """WATI message structure"""
    from_: Optional[str] = Field(None, alias="from")
    text: Optional[WATITextMessage] = None
    timestamp: Optional[str] = None

    class Config:
        populate_by_name = True


class WATIProfile(BaseModel):
    """WATI contact profile"""
    name: Optional[str] = "Friend"


class WATIContact(BaseModel):
    """WATI contact structure"""
    profile: Optional[WATIProfile] = None


class WATIWebhookPayload(BaseModel):
    """WATI webhook payload structure"""
    messages: List[WATIMessage] = []
    contacts: List[WATIContact] = []
    entry: List[Dict[str, Any]] = []


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def get_ai_response(
        industry: str,
        clinic_name: str,
        treatment_type: str,
        lead_name: str,
        conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Generate AI response using OpenAI GPT-4 with industry-specific persona

    Args:
        industry: 'dentistry' or 'dermatology'
        clinic_name: Name of the clinic
        treatment_type: Type of treatment the patient needs
        lead_name: Patient's name
        conversation_history: Previous conversation messages

    Returns:
        AI-generated response text
    """
    # Define industry-specific system prompts
    system_prompts = {
        "dentistry":
        f"You are Smile Buddy, a dental clinic assistant for {clinic_name}. "
        f"Help patients with {treatment_type}. Be warm and professional. "
        f"Keep responses friendly, concise, and encouraging. Focus on making patients "
        f"feel comfortable about their dental treatment.",
        "dermatology":
        f"You are Luna, a dermatology clinic assistant for {clinic_name}. "
        f"Help patients with skin concerns like {treatment_type}. "
        f"Be empathetic and professional. Show genuine care for their skin health. "
        f"Keep responses supportive and informative."
    }

    # Get the appropriate system prompt
    system_prompt = system_prompts.get(industry, system_prompts["dentistry"])

    # Build messages array
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)

    # Add initial greeting context
    messages.append({
        "role":
        "user",
        "content":
        f"Generate a warm greeting for {lead_name} who is interested in {treatment_type}. "
        f"Introduce yourself and offer to help them."
    })

    try:
        # Call OpenAI API
        response = await openai_client.chat.completions.create(
            model="gpt-4", messages=messages, temperature=0.7, max_tokens=150)

        return response.choices[0].message.content.strip()

    except Exception as e:
        # Log error and return fallback message
        logger.error(f"OpenAI API error: {str(e)}")
        bot_name = "Smile Buddy" if industry == "dentistry" else "Luna"
        return (
            f"Hi {lead_name}! I'm {bot_name} from {clinic_name}. "
            f"I'm here to help you with {treatment_type}. How can I assist you today?"
        )


def calculate_confidence_score(lead_data: Dict[str, Any]) -> float:
    """
    Calculate lead quality confidence score based on data completeness and urgency

    Args:
        lead_data: Dictionary containing lead information

    Returns:
        Confidence score between 0.0 and 1.0
    """
    score = 0.0

    # Base score for having complete required fields (0.3)
    if all(key in lead_data for key in ['name', 'phone', 'treatment_type']):
        score += 0.3

    # Phone number validation (0.2)
    phone = lead_data.get('phone', '')
    clean_phone = re.sub(r'[\s\-\(\)]+', '', phone)
    if re.match(r'^\+?[1-9]\d{9,14}$',
                clean_phone):  # Valid international format
        score += 0.2
    elif re.match(r'^\+?[1-9]\d{7,9}$', clean_phone):  # Basic valid format
        score += 0.1

    # Urgency level scoring (0.3)
    urgency = lead_data.get('treatment_urgency', '').lower()
    urgency_scores = {'urgent': 0.3, 'moderate': 0.20, 'not_urgent': 0.10}
    score += urgency_scores.get(urgency, 0.0)

    # Budget information provided (0.2)
    if lead_data.get('budget_range'):
        score += 0.2

    # Pain level for dermatology (bonus if high pain)
    pain_level = lead_data.get('pain_level', 0)
    if pain_level >= 7:
        score += 0.1
    elif pain_level >= 4:
        score += 0.05

    # Cap score at 1.0
    return min(score, 1.0)


def get_clinic_info(clinic_id: str) -> Dict[str, Any]:
    """
    Retrieve clinic information from database using clinic_id

    Args:
        clinic_id: Unique clinic identifier (TEXT field)

    Returns:
        Clinic information dictionary

    Raises:
        HTTPException: If clinic not found
    """
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Database error: {str(e)}")


# ============================================================================
# WHATSAPP MESSAGE HANDLING
# ============================================================================


async def send_message_to_whatsapp(wa_id: str, message: str) -> Dict[str, Any]:
    """
    Generic placeholder for sending WhatsApp messages.
    This function will be replaced with actual API integration (WATI, Meta, etc.)

    Args:
        wa_id: WhatsApp ID (phone number without +)
        message: Text message to send

    Returns:
        Dictionary with send status

    Note:
        Currently logs the message. Replace this function body with your
        chosen WhatsApp API provider (WATI, Meta Business API, Twilio, etc.)
    """
    logger.info(f"📤 [OUTBOUND MESSAGE]")
    logger.info(f"   To: +{wa_id}")
    logger.info(f"   Message: {message[:100]}...")

    # TODO: Integrate with your WhatsApp API provider here
    # Example integrations:
    # - WATI: POST to https://live-mt-server.wati.io/api/v1/sendSessionMessage/{wa_id}
    # - Meta: POST to https://graph.facebook.com/v18.0/{phone_number_id}/messages
    # - Twilio: POST to https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json

    return {
        "status": "logged",
        "wa_id": wa_id,
        "message_length": len(message),
        "note": "Message logged - integrate with WhatsApp API provider"
    }


async def process_whatsapp_message(
        lead_id: str,
        message_text: str,
        sender_wa_id: str,
        timestamp: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Core message processing logic: retrieves lead, generates AI response,
    updates conversation log, and sends reply.

    Args:
        lead_id: UUID of the lead in database
        message_text: Incoming message from user
        sender_wa_id: WhatsApp ID of sender
        timestamp: Message timestamp (defaults to now)

    Returns:
        Dictionary with processing results including AI response

    Raises:
        HTTPException: If lead not found or processing fails
    """
    try:
        # Retrieve lead from database
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

        # Add incoming message to conversation
        message_timestamp = (timestamp.isoformat() if timestamp else
                             datetime.now(timezone.utc).isoformat())

        conversation_log.append({
            "role": "user",
            "message": message_text,
            "timestamp": message_timestamp,
            "sender": sender_wa_id
        })

        # Get clinic info for context
        clinic = get_clinic_info(lead['clinic_id'])
        logger.info(
            f"🏥 Clinic: {clinic['clinic_name']} ({clinic['industry']})")

        # Prepare conversation history for AI (last 10 messages for context)
        recent_messages = [{
            "role": msg.get("role", "user"),
            "content": msg.get("message", "")
        } for msg in conversation_log[-10:]]

        # Generate AI response
        ai_response = await get_ai_response(
            industry=lead['industry'],
            clinic_name=clinic['clinic_name'],
            treatment_type=lead['treatment_type'],
            lead_name=lead['name'],
            conversation_history=recent_messages)

        logger.info(f"🤖 AI Response: {ai_response[:100]}...")

        # Add AI response to conversation log
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
            LeadStatus.CONTACTED  # Auto-update status to contacted
        }).eq('id', lead_id).execute()

        if not update_response.data:
            logger.error(f"❌ Failed to update lead {lead_id} in database")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update conversation log")

        # Send reply via WhatsApp (currently just logs)
        send_result = await send_message_to_whatsapp(wa_id=sender_wa_id,
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


async def find_or_create_lead(phone: str,
                              name: str,
                              clinic_id: str,
                              source: str = "whatsapp") -> Dict[str, Any]:
    """
    Find existing lead by phone or create new one.

    Args:
        phone: Phone number (with + prefix)
        name: Contact name
        clinic_id: Clinic identifier
        source: Lead source (default: whatsapp)

    Returns:
        Lead data dictionary with 'id' field

    Raises:
        HTTPException: If database operation fails
    """
    try:
        # Search for existing lead
        logger.info(f"🔍 Searching for lead with phone: {phone}")
        lead_resp = (supabase.table("leads").select("*").eq("phone", phone).eq(
            "clinic_id", clinic_id).limit(1).execute())

        if lead_resp.data:
            lead = lead_resp.data[0]
            logger.info(
                f"✅ Found existing lead: {lead['name']} (ID: {lead['id']})")
            return lead

        # Create new lead
        logger.info(f"➕ Creating new lead for {name} ({phone})")
        lead_data = {
            "clinic_id": clinic_id,
            "name": name,
            "phone": phone,
            "industry": "dentistry",  # Default, can be updated
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


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running

    Returns:
        Status dictionary
    """
    return {"status": "healthy"}


@app.post("/lead/dental",
          response_model=LeadResponse,
          status_code=status.HTTP_201_CREATED)
async def create_dental_lead(lead_request: DentalLeadRequest):
    """
    Create a new dental clinic lead with AI-generated greeting

    Args:
        lead_request: Dental lead information

    Returns:
        LeadResponse with lead details and AI greeting
    """
    try:
        # Get clinic information
        clinic = get_clinic_info(lead_request.clinic_id)

        # Ensure clinic is a dentistry clinic
        if clinic["industry"] != IndustryType.DENTISTRY:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="clinic_id is not a dentistry clinic")

        # Calculate confidence score
        lead_dict = lead_request.model_dump()
        confidence_score = calculate_confidence_score(lead_dict)

        # Generate AI greeting
        ai_greeting = await get_ai_response(
            industry="dentistry",
            clinic_name=clinic['clinic_name'],
            treatment_type=lead_request.treatment_type,
            lead_name=lead_request.name)

        # Create lead in database - let Postgres handle id, created_at, updated_at
        lead_data = {
            "clinic_id": lead_request.clinic_id,
            "name": lead_request.name,
            "phone": lead_request.phone,
            "industry": "dentistry",
            "treatment_type": lead_request.treatment_type,
            "treatment_urgency": lead_request.treatment_urgency,
            "budget_range": lead_request.budget_range,
            "status": LeadStatus.NEW,
            "confidence_score": confidence_score,
            "conversation_log": [{
                "role": "assistant",
                "message": ai_greeting
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to create dental lead: {str(e)}")


@app.post("/lead/dermatology",
          response_model=LeadResponse,
          status_code=status.HTTP_201_CREATED)
async def create_dermatology_lead(lead_request: DermatologyLeadRequest):
    """
    Create a new dermatology clinic lead with AI-generated greeting

    Args:
        lead_request: Dermatology lead information

    Returns:
        LeadResponse with lead details and AI greeting
    """
    try:
        # Get clinic information
        clinic = get_clinic_info(lead_request.clinic_id)

        # Ensure clinic is a dermatology clinic
        if clinic["industry"] != IndustryType.DERMATOLOGY:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="clinic_id is not a dermatology clinic")

        # Calculate confidence score (includes pain level)
        lead_dict = lead_request.model_dump()
        confidence_score = calculate_confidence_score(lead_dict)

        # Generate AI greeting
        ai_greeting = await get_ai_response(
            industry="dermatology",
            clinic_name=clinic['clinic_name'],
            treatment_type=lead_request.treatment_type,
            lead_name=lead_request.name)

        # Create lead in database - let Postgres handle id, created_at, updated_at
        lead_data = {
            "clinic_id": lead_request.clinic_id,
            "name": lead_request.name,
            "phone": lead_request.phone,
            "industry": "dermatology",
            "treatment_type": lead_request.treatment_type,
            "treatment_urgency": lead_request.treatment_urgency,
            "budget_range": lead_request.budget_range,
            "pain_level": lead_request.pain_level,
            "status": LeadStatus.NEW,
            "confidence_score": confidence_score,
            "conversation_log": [{
                "role": "assistant",
                "message": ai_greeting
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create dermatology lead: {str(e)}")


@app.get("/leads/{clinic_id}")
async def get_clinic_leads(clinic_id: str):
    """
    Retrieve all leads for a specific clinic

    Args:
        clinic_id: Unique clinic identifier

    Returns:
        List of leads for the clinic
    """
    try:
        # Verify clinic exists
        get_clinic_info(clinic_id)

        # Retrieve leads
        response = supabase.table('leads').select('*').eq(
            'clinic_id', clinic_id).order('created_at', desc=True).execute()

        return {"leads": response.data, "count": len(response.data)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to retrieve leads: {str(e)}")


@app.patch("/leads/{lead_id}")
async def update_lead(lead_id: str, lead_update: LeadUpdate):
    """
    Update lead status and/or confidence score

    Args:
        lead_id: Unique lead identifier (UUID)
        lead_update: Fields to update

    Returns:
        Updated lead information
    """
    try:
        # Check if lead exists
        lead_response = supabase.table('leads').select('*').eq(
            'id', lead_id).execute()

        if not lead_response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Lead with id {lead_id} not found")

        # Build update dictionary (only include provided fields)
        update_data = {}
        if lead_update.status is not None:
            update_data['status'] = lead_update.status
        if lead_update.confidence_score is not None:
            update_data['confidence_score'] = lead_update.confidence_score

        if not update_data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="No fields provided for update")

        # Update lead (updated_at handled by trigger)
        updated_response = supabase.table('leads').update(update_data).eq(
            'id', lead_id).execute()

        return {
            "lead": updated_response.data[0],
            "message": "Lead updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to update lead: {str(e)}")


@app.post("/clinic/register", status_code=status.HTTP_201_CREATED)
async def register_clinic(clinic: ClinicRegistration):
    """
    Register a new clinic in the system

    Args:
        clinic: Clinic registration information

    Returns:
        Clinic ID and registration details
    """
    try:
        # Generate unique clinic_id (TEXT field, not UUID)
        clinic_id = f"{clinic.industry.value}_{uuid.uuid4().hex[:12]}"

        # Determine bot name based on industry
        bot_name = "Smile Buddy" if clinic.industry == IndustryType.DENTISTRY else "Luna"

        # Create clinic record matching schema
        clinic_data = {
            "clinic_id": clinic_id,
            "clinic_name": clinic.clinic_name,
            "industry": clinic.industry,
            "owner_name": clinic.clinic_name,  # Temporary default
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to register clinic: {str(e)}")


@app.get("/clinic/{clinic_id}/dashboard")
async def get_clinic_dashboard(clinic_id: str):
    """
    Get dashboard statistics for a clinic

    Args:
        clinic_id: Unique clinic identifier

    Returns:
        Dashboard statistics including lead counts by status
    """
    try:
        # Verify clinic exists
        get_clinic_info(clinic_id)

        # Get all leads for the clinic
        response = supabase.table('leads').select('status').eq(
            'clinic_id', clinic_id).execute()

        # Count leads by status
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

        # Calculate conversion rate (booked + completed)
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dashboard data: {str(e)}")


@app.post("/webhook/wati/raw", status_code=status.HTTP_200_OK)
async def webhook_wati_raw(request: Request, payload: WATIWebhookPayload):
    """
    Universal WhatsApp webhook endpoint that accepts WATI-formatted payloads.
    Can be adapted for other WhatsApp providers with minimal changes.

    Workflow:
    1. Parse incoming WATI webhook payload
    2. Extract message, sender, and contact info
    3. Find or create lead in database
    4. Process message through AI
    5. Log outbound message (placeholder for actual send)

    Args:
        request: FastAPI request object (for debugging)
        payload: WATI webhook payload with messages and contacts

    Returns:
        Processing confirmation with AI response

    Raises:
        HTTPException: If payload invalid or processing fails
    """
    try:
        logger.info(f"📨 Received WhatsApp webhook from {request.client.host}")

        # ===== STEP 1: VALIDATE PAYLOAD =====
        if not payload.messages:
            logger.warning("⚠️ Webhook received with no messages")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Invalid payload: no messages found")

        message = payload.messages[0]

        # ===== STEP 2: EXTRACT SENDER INFO =====
        wa_id = message.from_
        if not wa_id:
            logger.error("❌ Missing 'from' field in webhook")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid payload: missing sender WhatsApp ID")

        # Normalize phone format
        phone = f"+{wa_id}" if not wa_id.startswith("+") else wa_id
        logger.info(f"👤 Sender: {phone}")

        # ===== STEP 3: EXTRACT CONTACT NAME =====
        name = "Friend"  # Default fallback
        if payload.contacts and payload.contacts[0].profile:
            name = payload.contacts[0].profile.name or "Friend"
        logger.info(f"👤 Name: {name}")

        # ===== STEP 4: EXTRACT MESSAGE TEXT =====
        text = ""
        if message.text and message.text.body:
            text = message.text.body.strip()

        if not text:
            logger.warning("⚠️ Received empty message")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Invalid payload: empty message text")

        logger.info(f"💬 Message: {text[:100]}...")

        # ===== STEP 5: EXTRACT TIMESTAMP =====
        timestamp = None
        if message.timestamp:
            try:
                timestamp = datetime.fromtimestamp(int(message.timestamp),
                                                   tz=timezone.utc)
            except (ValueError, TypeError) as e:
                logger.warning(f"⚠️ Invalid timestamp format: {e}")
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        # ===== STEP 6: DETERMINE CLINIC =====
        # TODO: Implement clinic routing logic based on phone number or webhook source
        DEFAULT_CLINIC_ID = "dental_clinic_001"

        # ===== STEP 7: FIND OR CREATE LEAD =====
        lead = await find_or_create_lead(phone=phone,
                                         name=name,
                                         clinic_id=DEFAULT_CLINIC_ID,
                                         source="whatsapp")

        lead_id = str(lead["id"])

        # ===== STEP 8: PROCESS MESSAGE & GENERATE RESPONSE =====
        result = await process_whatsapp_message(lead_id=lead_id,
                                                message_text=text,
                                                sender_wa_id=wa_id,
                                                timestamp=timestamp)

        logger.info(f"✅ Webhook processed successfully for lead {lead_id}")

        return {
            "success": True,
            "webhook_received": datetime.now(timezone.utc).isoformat(),
            "lead_id": lead_id,
            "lead_name": name,
            "message_received": text[:100],
            "ai_response": result["ai_response"],
            "processing_details": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected webhook error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Webhook processing failed: {str(e)}")


async def whatsapp_webhook(webhook_data: WhatsAppWebhook):
    """
    Legacy webhook handler - use webhook_wati_raw instead.
    Kept for backward compatibility with existing integrations.

    DEPRECATED: This function will be removed in future versions.
    Use the /webhook/wati/raw endpoint instead.
    """
    logger.warning("⚠️ Using deprecated whatsapp_webhook function")
    return await process_whatsapp_message(lead_id=webhook_data.lead_id,
                                          message_text=webhook_data.message,
                                          sender_wa_id=webhook_data.sender,
                                          timestamp=webhook_data.timestamp)


# ============================================================================
# STARTUP EVENT
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Medical Clinic Lead Generation API started successfully")
    logger.info("📋 Endpoints available at /docs")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    with open("dashboard_html.html", "r") as f:
        return f.read()


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
