from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import os
import sqlite3
import hashlib
import io
import json
from jose import JWTError, jwt
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Constants
DOCTOR_EMAIL = os.getenv("DOCTOR_EMAIL", "202411019@daiict.ac.in")
SECRET_KEY = os.getenv("SECRET_KEY", "yoursecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(title="DHARITRI - Digital Health Platform API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://192.168.1.7:8080",
    ],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 scheme for token-based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ---- Pydantic Models ----


class UserBase(BaseModel):
    username: str
    email: EmailStr
    role: str = "Patient"


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class ReportAnalysisRequest(BaseModel):
    user_id: str


class DietConsultationRequest(BaseModel):
    question: str
    report_text: Optional[str] = None


class ReportResponse(BaseModel):
    id: int
    user_id: str
    report_name: str
    upload_date: str
    analysis_result: str
    doctor_notes: Optional[str]
    doctor_approval: bool


class UpdateReportRequest(BaseModel):
    notes: str
    approval: bool


# ---- Database Functions ----


def init_database():
    """Initialize SQLite database with necessary tables"""
    conn = sqlite3.connect("medical_dashboard.db")
    c = conn.cursor()

    # User Reports Table
    c.execute(
        """CREATE TABLE IF NOT EXISTS user_reports
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  report_name TEXT,
                  upload_date DATETIME,
                  analysis_result TEXT,
                  doctor_notes TEXT,
                  doctor_approval INTEGER DEFAULT 0,
                  is_active INTEGER DEFAULT 1)"""
    )

    # User Table
    c.execute(
        """CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT,
                  email TEXT,
                  role TEXT)"""
    )

    conn.commit()
    conn.close()


def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def get_user(username: str):
    """Get user from database by username"""
    conn = sqlite3.connect("medical_dashboard.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()

    if user:
        return {
            "id": user[0],
            "username": user[1],
            "password": user[2],
            "email": user[3],
            "role": user[4],
        }
    return None


def authenticate_user(username: str, password: str):
    """Authenticate user with username and password"""
    user = get_user(username)
    if not user:
        return False
    if user["password"] != hash_password(password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def save_report_analysis(user_id: str, report_name: str, analysis_result: str):
    """Save report analysis to database"""
    conn = sqlite3.connect("medical_dashboard.db")
    c = conn.cursor()

    c.execute(
        """INSERT INTO user_reports 
                 (user_id, report_name, upload_date, analysis_result) 
                 VALUES (?, ?, ?, ?)""",
        (user_id, report_name, datetime.now(), analysis_result),
    )

    report_id = c.lastrowid
    conn.commit()
    conn.close()
    return report_id


def get_user_reports(user_id: str):
    """Get all reports for a user"""
    conn = sqlite3.connect("medical_dashboard.db")
    c = conn.cursor()

    c.execute(
        """SELECT * FROM user_reports 
                 WHERE user_id=? AND is_active=1 
                 ORDER BY upload_date DESC""",
        (user_id,),
    )
    reports = c.fetchall()
    conn.close()

    result = []
    for report in reports:
        result.append(
            {
                "id": report[0],
                "user_id": report[1],
                "report_name": report[2],
                "upload_date": report[3],
                "analysis_result": report[4],
                "doctor_notes": report[5],
                "doctor_approval": bool(report[6]),
            }
        )
    return result


def doctor_update_report(report_id: int, notes: str, approval: bool):
    """Update report with doctor notes and approval"""
    conn = sqlite3.connect("medical_dashboard.db")
    c = conn.cursor()

    c.execute(
        """UPDATE user_reports 
                 SET doctor_notes=?, doctor_approval=? 
                 WHERE id=?""",
        (notes, 1 if approval else 0, report_id),
    )

    conn.commit()
    conn.close()


def get_pending_reports():
    """Get all reports pending doctor approval"""
    conn = sqlite3.connect("medical_dashboard.db")
    c = conn.cursor()

    c.execute(
        """SELECT * FROM user_reports 
                 WHERE doctor_approval=0 
                 ORDER BY upload_date DESC"""
    )
    reports = c.fetchall()
    conn.close()

    result = []
    for report in reports:
        result.append(
            {
                "id": report[0],
                "user_id": report[1],
                "report_name": report[2],
                "upload_date": report[3],
                "analysis_result": report[4],
                "doctor_notes": report[5],
                "doctor_approval": bool(report[6]),
            }
        )
    return result


def get_all_patient_reports():
    """Get all patient reports with user information"""
    conn = sqlite3.connect("medical_dashboard.db")
    c = conn.cursor()

    c.execute(
        """SELECT 
            user_reports.id, 
            users.username, 
            users.email, 
            user_reports.report_name, 
            user_reports.upload_date, 
            user_reports.analysis_result, 
            user_reports.doctor_notes, 
            user_reports.doctor_approval 
        FROM user_reports 
        JOIN users ON user_reports.user_id = users.username 
        ORDER BY user_reports.upload_date DESC"""
    )
    reports = c.fetchall()
    conn.close()

    result = []
    for report in reports:
        result.append(
            {
                "id": report[0],
                "username": report[1],
                "email": report[2],
                "report_name": report[3],
                "upload_date": report[4],
                "analysis_result": report[5],
                "doctor_notes": report[6],
                "doctor_approval": bool(report[7]),
            }
        )
    return result


def register_user(username: str, password: str, email: str, role: str):
    """Register a new user"""
    conn = sqlite3.connect("medical_dashboard.db")
    c = conn.cursor()

    hashed_password = hash_password(password)
    try:
        c.execute(
            "INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)",
            (username, hashed_password, email, role),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


# ---- PDF and AI Processing Functions ----


def get_pdf_text(pdf_file):
    """Extract text from PDF file"""
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"
    return text


def get_text_chunks(text):
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    """Create and save vector store from text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_medical_analysis(text_chunks):
    """Generate medical analysis from text chunks"""
    prompt_template = """
You are a highly advanced AI medical assistant specializing in critical condition detection. Your task is to analyze structured patient data, including vital signs (heart rate, blood pressure, oxygen saturation, temperature), lab results, symptoms, and medical history, to identify patients in critical states.

Context:
{context}

Output Format:
- Doctor must see the following things
- Summarize data finding (show data and their conditions which do not match)
- Summarize Risk evaluation
- Food Recommendation
- Urgency to consult doctor
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    docs = new_db.similarity_search("Analyze the patient data for critical conditions.")

    response = chain({"input_documents": docs}, return_only_outputs=True)
    return response["output_text"]


def send_email(doctor_email, analysis_text, pdf_content, filename):
    """Send email with report analysis to doctor"""
    sender_email = os.getenv("SENDER_EMAIL", "ammarkarimi9898@gmail.com")
    sender_password = os.getenv("SENDER_PASSWORD", "nnep zuox lrif tcet")

    msg = EmailMessage()
    msg["Subject"] = "Patient Report Analysis"
    msg["From"] = sender_email
    msg["To"] = doctor_email
    msg.set_content(
        f"Dear Doctor,\n\nHere is the AI-generated medical report analysis:\n\n{analysis_text}\n\nBest Regards,\nAI Medical Assistant"
    )

    # Attach PDF
    msg.add_attachment(
        pdf_content, maintype="application", subtype="pdf", filename=filename
    )

    # Send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)


def ask_diet_question(input_question, report_text):
    """Generate response to diet-related question based on report"""
    # Step 1: Analyze the report for medical conditions
    analysis_prompt = f"""
You are a highly advanced AI medical assistant. Analyze the following medical report and identify any conditions, test results, or data that could influence dietary recommendations (e.g., diabetes, high cholesterol, hypertension, kidney issues, etc.).

Report:
{report_text}

Provide a concise summary of the key findings relevant to diet in one or two sentences.
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    analysis_response = model.invoke(analysis_prompt)
    analysis_summary = analysis_response.content

    # Step 2: Answer the user's diet-related question
    answer_prompt = f"""
You are an AI medical assistant specializing in dietary advice. Based on the analysis of a medical report, answer the user's diet-related question. If the report lacks specific dietary information, use the identified conditions or data to make an educated recommendation. If no relevant data is found, explain that and suggest consulting a doctor.

Report Analysis:
{analysis_summary}

User Question:
{input_question}

Provide a clear, concise answer to the user's question, focusing on whether the requested food or diet is suitable based on the report analysis.
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    response = model.invoke(answer_prompt)
    return response.content


# ---- Authentication Dependency ----


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Verify JWT token and return current user"""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_doctor(current_user: dict = Depends(get_current_user)):
    """Check if current user is a doctor"""
    if current_user["role"] != "Doctor":
        raise HTTPException(status_code=403, detail="Not authorized")
    return current_user


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_database()


# ---- Authentication Endpoints ----


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Endpoint for user authentication and token generation"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/register", status_code=201)
async def create_user(user: UserCreate):
    """Endpoint to register a new user"""
    if register_user(user.username, user.password, user.email, user.role):
        return {"message": "User registered successfully"}
    else:
        raise HTTPException(status_code=400, detail="Username already exists")


# ---- Patient Endpoints ----


@app.post("/reports/analyze")
async def analyze_report(
    files: List[UploadFile] = File(...), current_user: dict = Depends(get_current_user)
):
    """Endpoint to analyze medical reports"""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Process first PDF file
    pdf_file = files[0]
    pdf_content = await pdf_file.read()
    pdf_io = io.BytesIO(pdf_content)

    # Extract text from PDF
    raw_text = get_pdf_text(pdf_io)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    # Generate analysis
    analysis_result = get_medical_analysis(text_chunks)

    # Save analysis to database
    report_id = save_report_analysis(
        current_user["username"], pdf_file.filename, analysis_result
    )

    # Send email to doctor
    send_email(DOCTOR_EMAIL, analysis_result, pdf_content, pdf_file.filename)

    return {
        "report_id": report_id,
        "analysis": analysis_result,
        "message": f"Report sent to {DOCTOR_EMAIL}",
    }


@app.post("/diet/consult")
async def diet_consultation(
    request: DietConsultationRequest, current_user: dict = Depends(get_current_user)
):
    """Endpoint for diet consultation questions"""
    # Use provided report text or generic context
    report_context = request.report_text or "No specific medical report uploaded."

    # Generate response
    response = ask_diet_question(request.question, report_context)

    return {"question": request.question, "answer": response}


@app.get("/reports", response_model=List[Dict[str, Any]])
async def get_reports(current_user: dict = Depends(get_current_user)):
    """Endpoint to get all reports for the current user"""
    reports = get_user_reports(current_user["username"])
    return reports


# ---- Doctor Endpoints ----


@app.get("/reports/pending", response_model=List[Dict[str, Any]])
async def get_pending_reports_endpoint(
    current_user: dict = Depends(get_current_doctor),
):
    """Endpoint to get all reports pending approval"""
    reports = get_pending_reports()
    return reports


@app.put("/reports/{report_id}")
async def update_report(
    report_id: int,
    request: UpdateReportRequest,
    current_user: dict = Depends(get_current_doctor),
):
    """Endpoint to update report with doctor notes and approval"""
    doctor_update_report(report_id, request.notes, request.approval)
    return {"message": "Report updated successfully"}


@app.get("/reports/all", response_model=List[Dict[str, Any]])
async def get_all_reports(
    current_user: dict = Depends(get_current_doctor),
    name_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Endpoint to get all patient reports with filtering options"""
    reports = get_all_patient_reports()

    # Apply filters if provided
    filtered_reports = reports

    if name_filter:
        filtered_reports = [
            report
            for report in filtered_reports
            if name_filter.lower() in report["username"].lower()
        ]

    if start_date:
        start_datetime = datetime.fromisoformat(start_date)
        filtered_reports = [
            report
            for report in filtered_reports
            if datetime.strptime(report["upload_date"], "%Y-%m-%d %H:%M:%S.%f")
            >= start_datetime
        ]

    if end_date:
        end_datetime = datetime.fromisoformat(end_date)
        filtered_reports = [
            report
            for report in filtered_reports
            if datetime.strptime(report["upload_date"], "%Y-%m-%d %H:%M:%S.%f")
            <= end_datetime
        ]

    return filtered_reports


# Basic health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}
