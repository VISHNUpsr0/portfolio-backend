# backend/spamclassiferapi.py

import os
from pathlib import Path
from typing import Literal

import joblib
import uvicorn

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from groq import Groq


# =========================================================
# PATHS + ENVIRONMENT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


# =========================================================
# LOAD SPAM MODEL
# =========================================================

MODEL_PATH = BASE_DIR / "spam_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Spam model not found: {MODEL_PATH}"
    )

spam_model = joblib.load(MODEL_PATH)

print("Spam classifier model loaded successfully")


# =========================================================
# GROQ CLIENT
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:
    groq_client = Groq(
        api_key=GROQ_API_KEY
    )

    print("Groq client initialized successfully")

else:
    groq_client = None

    print(
        "WARNING: GROQ_API_KEY not found. "
        "Spam classifier will work, but chatbot will not."
    )


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Vishnu AI Portfolio API",
    description=(
        "Backend API for the portfolio spam classifier "
        "and AI portfolio assistant."
    ),
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST MODELS
# =========================================================

class EmailRequest(BaseModel):
    email: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str

    history: list[ChatMessage] = Field(
        default_factory=list
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/")
def root():
    return {
        "message": "Vishnu Portfolio API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "spam_model": "loaded",
        "chatbot": (
            "ready"
            if groq_client
            else "GROQ_API_KEY missing"
        )
    }


# =========================================================
# SPAM CLASSIFIER
# =========================================================

@app.post("/predict")
def predict_spam(request: EmailRequest):

    email_text = request.email.strip()

    if not email_text:
        raise HTTPException(
            status_code=400,
            detail="Email cannot be empty."
        )


    try:

        prediction = spam_model.predict(
            [email_text]
        )[0]


        # -----------------------------------------
        # Convert prediction into True / False
        # Supports:
        # 1 / 0
        # "spam" / "ham"
        # True / False
        # -----------------------------------------

        prediction_text = (
            str(prediction)
            .strip()
            .lower()
        )


        is_spam = prediction_text in {
            "1",
            "spam",
            "true"
        }


        # -----------------------------------------
        # Confidence
        # -----------------------------------------

        confidence = None


        if hasattr(spam_model, "predict_proba"):

            probabilities = spam_model.predict_proba(
                [email_text]
            )[0]


            classes = getattr(
                spam_model,
                "classes_",
                None
            )


            if classes is not None:

                classes_list = list(classes)

                try:

                    predicted_index = (
                        classes_list.index(
                            prediction
                        )
                    )

                    confidence = float(
                        probabilities[
                            predicted_index
                        ]
                    )

                except ValueError:

                    confidence = float(
                        max(probabilities)
                    )

            else:

                confidence = float(
                    max(probabilities)
                )


        return {
            "is_spam": is_spam,
            "prediction": (
                "spam"
                if is_spam
                else "ham"
            ),
            "confidence": confidence
        }


    except Exception as error:

        print(
            "Spam prediction error:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail="Spam prediction failed."
        )
    # =========================================================
# PORTFOLIO AI ASSISTANT
# =========================================================

PORTFOLIO_SYSTEM_PROMPT = """
You are the AI portfolio assistant for Vishnu P.

Your purpose is to answer questions about Vishnu's
projects, technical skills, education and professional
background.

PROFILE

Name:
Vishnu P

Education:
B.Tech in Computer Science and Engineering
College of Engineering Trivandrum.

Career focus:
AI/ML Engineering
Python Development
Backend Development


PROJECTS

1. DocuChat

A Retrieval-Augmented Generation (RAG)
document assistant.

Technologies include:
Python
FastAPI
LangChain
FAISS
LLMs
React


2. CI/CD Log Debugging Assistant

An AI-assisted debugging system designed to analyze
CI/CD failure logs, identify possible root causes and
generate remediation suggestions.

The project includes work involving:
Mistral-7B
LoRA fine-tuning
RAG
Python
FastAPI


3. Real-Time Object Detection and Navigation
for Visually Impaired Users

A computer vision application using YOLOv8
to detect surrounding objects and provide useful
feedback to visually impaired users.

Technologies include:
YOLOv8
Python
OpenCV
Flutter


4. Spam Classifier

A machine-learning email spam classification system.

Technologies:
TF-IDF
Random Forest
Scikit-learn
FastAPI
React

Reported model accuracy:
97.85%


5. House Price Predictor

A machine-learning regression project for
predicting house prices.

Reported R²:
approximately 0.80


SKILLS

Python
FastAPI
Machine Learning
Deep Learning
Scikit-learn
PyTorch
TensorFlow
LLMs
RAG
LangChain
LangGraph
FAISS
LoRA
YOLOv8
Computer Vision
SQL
Docker
React
JavaScript
Git
GitHub


CONTACT

GitHub:
github.com/VISHNUpsr0

LinkedIn:
linkedin.com/in/vishnu-p-128520202

Email:
vishnupsr0@gmail.com


RESPONSE RULES

Keep answers concise and professional.

Normally respond in 2 to 4 sentences.

Only answer questions related to Vishnu's
portfolio, technical skills, projects,
education, career or contact information.

Never invent qualifications, work experience,
employment history, project results or technical
details that are not provided above.

If information is unavailable, say that you
do not have that information.

When someone asks how to contact Vishnu,
provide his email, LinkedIn or GitHub.
"""


@app.post("/chat")
def chat(request: ChatRequest):

    if groq_client is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "GROQ_API_KEY is not configured."
            )
        )


    current_message = (
        request.message.strip()
    )


    if not current_message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )


    messages = [
        {
            "role": "system",
            "content": PORTFOLIO_SYSTEM_PROMPT
        }
    ]


    # Only send recent history.
    # Prevents the prompt from growing forever.

    recent_history = request.history[-10:]


    for message in recent_history:

        messages.append(
            {
                "role": message.role,
                "content": message.content
            }
        )


    messages.append(
        {
            "role": "user",
            "content": current_message
        }
    )


    try:

        response = (
            groq_client
            .chat
            .completions
            .create(
                model=(
                    "openai/gpt-oss-120b"
                ),
                messages=messages,
                max_tokens=250,
                temperature=0.3
            )
        )


        reply = (
            response
            .choices[0]
            .message
            .content
        )


        return {
            "reply": reply,
            "role": "assistant"
        }


    except Exception as error:

        print(
            "Groq API error:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "AI assistant is temporarily unavailable."
            )
        )


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )