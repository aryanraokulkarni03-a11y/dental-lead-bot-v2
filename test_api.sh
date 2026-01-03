#!/bin/bash
API_URL="http://localhost:8000"

echo "========================================="
echo "Testing Medical Clinic Lead API"
echo "========================================="

# Test 1: Health Check
echo -e "\n1. Health Check"
curl -s "$API_URL/health" | python3 -m json.tool

# Test 2: Register a Clinic
echo -e "\n\n2. Registering Test Clinic..."
CLINIC_RESPONSE=$(curl -s -X POST "$API_URL/clinic/register" \
  -H "Content-Type: application/json" \
  -d '{
    "clinic_name": "Smile Dental Care",
    "industry": "dentistry",
    "contact_email": "admin@smiledentalcare.com",
    "contact_phone": "+919876543210"
  }')
echo "$CLINIC_RESPONSE" | python3 -m json.tool

# Extract clinic_id
CLINIC_ID=$(echo "$CLINIC_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['clinic_id'])" 2>/dev/null)
echo "Clinic ID: $CLINIC_ID"

# Test 3: Create a Dental Lead
echo -e "\n\n3. Creating Dental Lead..."
LEAD_RESPONSE=$(curl -s -X POST "$API_URL/lead/dental" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"Rahul Sharma\",
    \"phone\": \"+919876543211\",
    \"clinic_id\": \"$CLINIC_ID\",
    \"treatment_type\": \"Root Canal\",
    \"treatment_urgency\": \"urgent\",
    \"budget_range\": \"20k_50k\"
  }")
echo "$LEAD_RESPONSE" | python3 -m json.tool

LEAD_ID=$(echo "$LEAD_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['lead_id'])" 2>/dev/null)
echo "Lead ID: $LEAD_ID"

# Test 4: Get All Leads for Clinic
echo -e "\n\n4. Getting All Leads for Clinic..."
curl -s "$API_URL/leads/$CLINIC_ID" | python3 -m json.tool

# Test 5: Get Dashboard Stats
echo -e "\n\n5. Getting Dashboard Stats..."
curl -s "$API_URL/clinic/$CLINIC_ID/dashboard" | python3 -m json.tool

# Test 6: Update Lead Status
echo -e "\n\n6. Updating Lead Status..."
curl -s -X PATCH "$API_URL/leads/$LEAD_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "contacted",
    "confidence_score": 0.85
  }' | python3 -m json.tool

echo -e "\n\n========================================="
echo "All Tests Complete!"
echo "========================================="