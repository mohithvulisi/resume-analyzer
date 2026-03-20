from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pypdf import PdfReader
import io, json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

llm = ChatGroq(model="llama-3.3-70b-versatile")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), job_role: str = Form(...)):
    contents = await file.read()
    reader = PdfReader(io.BytesIO(contents))
    resume_text = ""
    for page in reader.pages:
        resume_text += page.extract_text()

    prompt = f"""
You are an expert resume analyzer and career coach.
Analyze this resume for the job role: "{job_role}"

Resume:
{resume_text[:4000]}

Respond ONLY with a valid JSON object like this:
{{
  "score": 78,
  "summary": "Brief 2-line overall summary",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "weaknesses": ["weakness 1", "weakness 2", "weakness 3"],
  "missing_keywords": ["keyword1", "keyword2", "keyword3", "keyword4"],
  "ats_score": 65,
  "top_suggestion": "The single most impactful change to make"
}}

Only return the JSON, nothing else.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        text = response.content.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        result = json.loads(text)
    except:
        result = {
            "score": 0,
            "summary": "Could not parse resume properly.",
            "strengths": [],
            "weaknesses": [],
            "missing_keywords": [],
            "ats_score": 0,
            "top_suggestion": "Please try again with a clearer PDF."
        }

    return result


@app.post("/suggest-roles")
async def suggest_roles(file: UploadFile = File(...)):
    contents = await file.read()
    reader = PdfReader(io.BytesIO(contents))
    resume_text = ""
    for page in reader.pages:
        resume_text += page.extract_text()

    prompt = f"""
You are an expert career coach and talent advisor.
Read this resume carefully and suggest the best job roles for this person based on their actual skills, experience, and education.

Resume:
{resume_text[:4000]}

Respond ONLY with a valid JSON object like this:
{{
  "name": "Person's name if found, else 'You'",
  "top_skills": ["skill1", "skill2", "skill3", "skill4", "skill5"],
  "roles": [
    {{
      "title": "Job Role Title",
      "match": 92,
      "reason": "One sentence why this role fits them",
      "salary": "Expected salary range in INR per year"
    }},
    {{
      "title": "Job Role Title",
      "match": 85,
      "reason": "One sentence why this role fits them",
      "salary": "Expected salary range in INR per year"
    }},
    {{
      "title": "Job Role Title",
      "match": 78,
      "reason": "One sentence why this role fits them",
      "salary": "Expected salary range in INR per year"
    }},
    {{
      "title": "Job Role Title",
      "match": 70,
      "reason": "One sentence why this role fits them",
      "salary": "Expected salary range in INR per year"
    }}
  ],
  "career_tip": "One powerful personalized career advice sentence for this person"
}}

Only return the JSON, nothing else.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        text = response.content.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        result = json.loads(text)
    except:
        result = {
            "name": "You",
            "top_skills": [],
            "roles": [],
            "career_tip": "Please try again with a clearer PDF."
        }

    return result


@app.get("/")
async def root():
    return FileResponse("index.html")