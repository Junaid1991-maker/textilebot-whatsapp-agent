# src/api/main.py
# FastAPI backend for TextileBot WhatsApp Agent
# Fixed for Azure: ChromaDB removed, replaced with keyword search

import os
import sys
import logging
import json
import uuid
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables first
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# AI imports
from groq import Groq
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict

# ── Rate Limiter Setup ──────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── FastAPI App Setup ───────────────────────────────────────────────────────
app = FastAPI(
    title="TextileBot API",
    description="WhatsApp AI Agent for B2B Textile Businesses",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Pydantic Models ─────────────────────────────────────────────────────────

class MessageRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    buyer_message: str = Field(..., min_length=1, max_length=1000)
    conversation_history: List[Dict] = Field(default_factory=list)

    @validator("buyer_message")
    def sanitize_message(cls, v):
        v = re.sub(r"<[^>]+>", "", v)
        injection_patterns = [
            r"ignore.{0,20}(instructions|prompts|rules)",
            r"system prompt",
            r"forget.{0,20}(everything|instructions)",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Invalid input detected")
        return v.strip()


class MessageResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    lead_score: int
    lead_status: str
    call_booked: bool
    escalated: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    rag_chunks: int


# ── Agent State ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    message: str
    conversation_history: List[Dict]
    intent: str
    intent_confidence: float
    buyer_info: Dict
    lead_score: int
    lead_status: str
    rag_context: str
    response: str
    call_booked: bool
    escalated: bool
    session_id: str


# ── Global Variables ────────────────────────────────────────────────────────

knowledge_base = {}   # filename -> full text content
agent = None
groq_api_key = None


# ── Keyword Search (replaces ChromaDB) ─────────────────────────────────────

def load_knowledge_base():
    """Load all .txt files from knowledge_base into memory."""
    global knowledge_base
    kb_path = Path(__file__).parent.parent.parent / "data" / "knowledge_base"
    knowledge_base = {}
    for txt_file in kb_path.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            knowledge_base[txt_file.name] = f.read()
    logger.info(f"Loaded {len(knowledge_base)} knowledge base files")


def retrieve_context(query: str, num_chunks: int = 3) -> str:
    """
    Keyword search over knowledge base files.
    Scores each 400-char chunk by how many query words it contains.
    Returns top num_chunks results.
    """
    query_words = set(query.lower().split())
    scored_chunks = []

    for filename, content in knowledge_base.items():
        # Split into overlapping chunks of ~400 chars
        words = content.split()
        chunk_size = 60  # words per chunk
        step = 40        # overlap
        for i in range(0, max(1, len(words) - chunk_size + 1), step):
            chunk = " ".join(words[i:i + chunk_size])
            chunk_lower = chunk.lower()
            score = sum(1 for w in query_words if w in chunk_lower)
            if score > 0:
                scored_chunks.append((score, filename, chunk))

    # Sort by score descending, take top N
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top = scored_chunks[:num_chunks]

    if not top:
        # Fallback: return first chunk of each file
        parts = []
        for filename, content in list(knowledge_base.items())[:2]:
            parts.append(f"[Source: {filename}]\n{content[:400]}")
        return "\n\n".join(parts)

    parts = [f"[Source: {filename}]\n{chunk}" for _, filename, chunk in top]
    return "\n\n".join(parts)


# ── Agent Node Functions ────────────────────────────────────────────────────

def node_intent_classifier(state: AgentState) -> AgentState:
    client = Groq(api_key=groq_api_key)
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """Classify buyer message into ONE of:
GREETING, PRODUCT_INQUIRY, PRICE_INQUIRY, CERTIFICATION_QUERY,
BOOKING_REQUEST, DOCUMENT_REQUEST, COMPLAINT, SPAM, COMPLEX

Return ONLY JSON: {"intent": "INTENT", "confidence": 0.95, "reasoning": "reason"}"""
                },
                {"role": "user", "content": f"Classify: {state['message']}"}
            ],
            max_tokens=100,
            timeout=30,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        state["intent"] = result["intent"]
        state["intent_confidence"] = result["confidence"]
    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        state["intent"] = "COMPLEX"
        state["intent_confidence"] = 0.5
    return state


def node_rag_retriever(state: AgentState) -> AgentState:
    state["rag_context"] = retrieve_context(state["message"])
    return state


def node_lead_qualifier(state: AgentState) -> AgentState:
    client = Groq(api_key=groq_api_key)
    history_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in state["conversation_history"]
    ])
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """Extract buyer info from conversation.
Return ONLY JSON:
{"buyer_name":"name or unknown","company":"company or unknown",
"country":"country or unknown","product_interest":"product or unknown",
"quantity":"quantity or unknown","certification":"cert or none",
"timeline":"timeline or unknown","budget":"budget or unknown",
"qualification_complete": true/false}"""
                },
                {"role": "user", "content": f"Extract from:\n{history_text}"}
            ],
            max_tokens=250,
            timeout=30,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        buyer_info = json.loads(text)
        state["buyer_info"] = buyer_info

        score = 0
        quantity_str = str(buyer_info.get("quantity", ""))
        numbers = re.findall(r'\d+', quantity_str)
        if numbers:
            qty = int(numbers[0])
            if qty >= 5000:
                score += 30
            elif qty >= 1000:
                score += 15
            elif qty > 0:
                score += 5

        country = str(buyer_info.get("country", "")).lower()
        premium = ["germany", "uk", "united kingdom", "united states",
                   "usa", "france", "italy", "netherlands", "sweden",
                   "denmark", "belgium", "canada", "australia", "japan"]
        if any(m in country for m in premium):
            score += 20
        elif country and country != "unknown":
            score += 10

        cert = str(buyer_info.get("certification", "")).lower()
        if cert and cert not in ["none", "unknown", ""]:
            score += 15

        timeline = str(buyer_info.get("timeline", "")).lower()
        if timeline and timeline not in ["unknown", ""]:
            score += 15

        budget = str(buyer_info.get("budget", "")).lower()
        if budget and budget not in ["unknown", ""]:
            score += 20

        state["lead_score"] = score
        state["lead_status"] = "HOT" if score >= 80 else "WARM" if score >= 50 else "COLD"

    except Exception as e:
        logger.error(f"Lead qualification error: {e}")
        state["buyer_info"] = {}
        state["lead_score"] = 0
        state["lead_status"] = "COLD"

    return state


def node_response_generator(state: AgentState) -> AgentState:
    client = Groq(api_key=groq_api_key)

    if state["lead_status"] == "HOT":
        closing = "This is a HOT lead. Offer to book a discovery call and include: https://calendly.com/textilebot/discovery"
        state["call_booked"] = True
    elif state["lead_status"] == "WARM":
        closing = "Offer to send the product catalogue."
        state["call_booked"] = False
    else:
        closing = "Be helpful and professional."
        state["call_booked"] = False

    history_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in state["conversation_history"][-6:]
    ])

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are TextileBot for a Pakistani textile export company.

CONTEXT:
{state["rag_context"]}

RULES:
- Answer only from context
- Be professional and concise
- {closing}"""
                },
                {
                    "role": "user",
                    "content": f"Conversation:\n{history_text}\n\nLatest: {state['message']}\n\nRespond:"
                }
            ],
            max_tokens=400,
            timeout=30,
        )
        state["response"] = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        state["response"] = "Thank you for your message. Our team will respond shortly."

    return state


def node_complaint_handler(state: AgentState) -> AgentState:
    state["escalated"] = True
    state["response"] = (
        "I sincerely apologize for the inconvenience. "
        "I am escalating your concern to our senior team immediately. "
        "A team member will contact you within 2 hours. "
        "Please share your order reference number so we can prioritize your case."
    )
    return state


def node_spam_filter(state: AgentState) -> AgentState:
    state["response"] = ""
    return state


def route_after_intent(state: AgentState) -> str:
    if state["intent"] == "COMPLAINT":
        return "complaint_handler"
    elif state["intent"] == "SPAM":
        return "spam_filter"
    else:
        return "rag_retriever"


def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("intent_classifier", node_intent_classifier)
    graph.add_node("rag_retriever", node_rag_retriever)
    graph.add_node("lead_qualifier", node_lead_qualifier)
    graph.add_node("response_generator", node_response_generator)
    graph.add_node("complaint_handler", node_complaint_handler)
    graph.add_node("spam_filter", node_spam_filter)

    graph.set_entry_point("intent_classifier")

    graph.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "rag_retriever": "rag_retriever",
            "complaint_handler": "complaint_handler",
            "spam_filter": "spam_filter",
        }
    )

    graph.add_edge("rag_retriever", "lead_qualifier")
    graph.add_edge("lead_qualifier", "response_generator")
    graph.add_edge("response_generator", END)
    graph.add_edge("complaint_handler", END)
    graph.add_edge("spam_filter", END)

    return graph.compile()


# ── Startup Event ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global agent, groq_api_key

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY not found")
        raise RuntimeError("GROQ_API_KEY not set")

    logger.info("Loading knowledge base...")
    load_knowledge_base()

    logger.info("Building agent...")
    agent = build_agent()

    logger.info("TextileBot API ready")


# ── API Endpoints ───────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        rag_chunks=len(knowledge_base)
    )


@app.post("/respond", response_model=MessageResponse)
@limiter.limit("10/minute")
async def respond(request: Request, message_request: MessageRequest):
    global agent

    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    conversation_history = message_request.conversation_history.copy()
    conversation_history.append({
        "role": "buyer",
        "content": message_request.buyer_message
    })

    initial_state: AgentState = {
        "message": message_request.buyer_message,
        "conversation_history": conversation_history,
        "intent": "",
        "intent_confidence": 0.0,
        "buyer_info": {},
        "lead_score": 0,
        "lead_status": "COLD",
        "rag_context": "",
        "response": "",
        "call_booked": False,
        "escalated": False,
        "session_id": message_request.session_id
    }

    try:
        final_state = agent.invoke(initial_state)

        logger.info(
            f"Session {message_request.session_id} - "
            f"Intent: {final_state['intent']} - "
            f"Score: {final_state['lead_score']} - "
            f"Status: {final_state['lead_status']}"
        )

        return MessageResponse(
            session_id=message_request.session_id,
            response=final_state["response"],
            intent=final_state["intent"],
            lead_score=final_state["lead_score"],
            lead_status=final_state["lead_status"],
            call_booked=final_state["call_booked"],
            escalated=final_state["escalated"]
        )

    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail="Agent processing failed")


@app.get("/leads")
async def get_leads():
    return {
        "message": "Lead data endpoint - connect to Google Sheets in production",
        "status": "ok"
    }


# ── Run the API ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )