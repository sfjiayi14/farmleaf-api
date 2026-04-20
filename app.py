import google.generativeai as genai
from PIL import Image
import json
import io
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for Firebase Studio to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔑 Read API key from environment variable (SECURE - not hardcoded)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# PERSON 1'S PROMPT (fine-tuned or base model)
SYSTEM_PROMPT = """
You are a Rice Paddy Pathologist.

DIAGNOSTIC RULES:
1. Rice Blast (Magnaporthe oryzae) — The "Eye" Pattern
Lesion Shape: Always look for Spindle-shaped, Diamond-shaped, or Eye-shaped spots. They are widest in the middle and pointed at both ends.
Internal Detail: Lesions have a distinct pale grayish-white to tan center (resembling an "eye").
The Border: Every lesion has a sharp, dark reddish-brown to dark brown margin or "halo."
Placement: Spots are scattered randomly across the entire width of the leaf blade, not restricted to the edges.
Merging: When severe, individual "eyes" merge (coalesce) into large irregular brown patches, but you can usually still see the "pointed ends" of the spindles at the edges of the dead tissue.

2. Bacterial Leaf Blight (Xanthomonas oryzae) — The "Streak" Pattern
Lesion Shape: Look for long, continuous longitudinal streaks rather than individual spots.
Starting Point: The infection almost always begins at the leaf tip or along the leaf margins (edges) and moves downward.
Internal Detail: Lesions are a uniform yellowish, straw-colored, or pale white color. There is no "center" versus "border."
The Boundary: The line between the green healthy tissue and the diseased straw-colored tissue is typically wavy or irregular, following the veins.
Placement: The disease is vascular, meaning it follows the "tracks" of the leaf veins.
Advanced Stage: Entire leaf blades (or large one-sided sections) turn completely straw-colored and dry out ("leaf firing") in a straight, longitudinal path.

OUTPUT RULE: You must ONLY respond in valid JSON format. Do not include conversational filler or markdown formatting like ```json.

JSON SCHEMA:
{
 "disease": "Rice Blast | Bacterial Leaf Blight | Healthy | Unknown",
 "confidence_score": "0-100%",
 "severity": "Low | Medium | High",
 "treatment": {
   "organic": ["step 1", "step 2"],
   "chemical": ["step 1", "step 2"]
 },
 "prevention": ["tip 1", "tip 2"]
}
"""

# Use base model (or fine-tuned model if Person 1 provides one)
# If using fine-tuned model, change to: model_name="tunedModels/your-model-name"
model = genai.GenerativeModel(
    model_name="gemini-3-flash-preview",
    system_instruction=SYSTEM_PROMPT
)

def send_email_alert(farmer_name, farmer_email, farmer_phone, disease, severity, treatment):
    """Send email alert to farmer (demo mode - prints instead of actually sending)"""
    
    print("\n" + "="*50)
    print("📧 EMAIL ALERT TRIGGERED")
    print("="*50)
    print(f"To: {farmer_name} <{farmer_email}>")
    print(f"Phone: {farmer_phone}")
    print(f"Subject: 🌾 PLANT DISEASE ALERT - {disease}")
    print(f"\nDear {farmer_name},")
    print(f"\nOur AI system has detected {disease} on your crop.")
    print(f"Severity: {severity}")
    print(f"Confidence: High")
    print(f"\n📋 TREATMENT RECOMMENDATIONS:")
    print(f"  Organic: {treatment.get('organic', ['N/A'])[0]}")
    print(f"  Chemical: {treatment.get('chemical', ['N/A'])[0]}")
    print(f"\n🛡️ PREVENTION TIPS:")
    for tip in treatment.get('prevention', ['Monitor crops regularly']):
        print(f"  - {tip}")
    print("\n" + "="*50)
    print("✅ Email sent successfully (Demo Mode)")
    print("="*50)
    
    return {
        "action": "email_sent",
        "recipient": farmer_email,
        "subject": f"Plant Disease Alert - {disease}",
        "message": f"Email sent to {farmer_name} about {disease} (Severity: {severity})"
    }

# API Endpoint for Firebase Studio
@app.post("/predict")
async def predict(
    firstName: str = Form(...),
    lastName: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
):
    """
    Endpoint that accepts form data + image from Firebase Studio
    Returns diagnosis and sends email alert
    """
    
    try:
        # Read the uploaded image
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data))
        
        # Get diagnosis from Gemini
        response = model.generate_content([img, "Analyze this rice leaf for diseases."])
        
        # Clean and parse JSON response
        cleaned_response = response.text.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        
        result = json.loads(cleaned_response)
        
        disease = result.get("disease", "Unknown")
        confidence = result.get("confidence_score", "0%")
        severity = result.get("severity", "Low")
        treatment = result.get("treatment", {})
        prevention = result.get("prevention", [])
        
        # ALWAYS send email alert (regardless of severity)
        farmer_name = f"{firstName} {lastName}"
        action_result = send_email_alert(farmer_name, email, phone, disease, severity, treatment)
        
        # Return JSON response for the website
        return {
            "success": True,
            "farmer": {
                "name": farmer_name,
                "email": email,
                "phone": phone
            },
            "diagnosis": {
                "disease": disease,
                "confidence": confidence,
                "severity": severity,
                "description": description or "No additional description provided"
            },
            "treatment": treatment,
            "prevention": prevention,
            "action_taken": action_result,
            "message": f"✅ {disease} detected. Email alert sent to {email}."
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to analyze image. Please try again."
        }

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "healthy", "service": "FarmLeaf Disease Detector API"}

# Run the server (for local testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)