# streamlit_app/app.py
# TextileBot — WhatsApp AI Agent Demo
# Refactored for Render.com: calls LangGraph agent directly (no FastAPI)

import streamlit as st
import json
import os
import re
import uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="TextileBot — WhatsApp AI Agent Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #f0f2f6; }
.buyer-message {
    background-color: #dcf8c6;
    padding: 10px 15px;
    border-radius: 15px 15px 0px 15px;
    margin: 5px 0;
    max-width: 80%;
    float: right;
    clear: both;
    color: #000;
}
.agent-message {
    background-color: #ffffff;
    padding: 10px 15px;
    border-radius: 15px 15px 15px 0px;
    margin: 5px 0;
    max-width: 80%;
    float: left;
    clear: both;
    color: #000;
}
.clearfix { clear: both; }
</style>
""", unsafe_allow_html=True)

# ── Import Agent ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from groq import Groq
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict

# ── Knowledge Base Loader ────────────────────────────────────────────────────
@st.cache_resource
def load_knowledge_base():
    kb_path = Path(__file__).parent.parent / "data" / "knowledge_base"
    kb = {}
    for txt_file in kb_path.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            kb[txt_file.name] = f.read()
    return kb

def retrieve_context(query: str, kb: dict, num_chunks: int = 3) -> str:
    query_words = set(query.lower().split())
    scored_chunks = []
    for filename, content in kb.items():
        words = content.split()
        chunk_size = 60
        step = 40
        for i in range(0, max(1, len(words) - chunk_size + 1), step):
            chunk = " ".join(words[i:i + chunk_size])
            score = sum(1 for w in query_words if w in chunk.lower())
            if score > 0:
                scored_chunks.append((score, filename, chunk))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top = scored_chunks[:num_chunks]
    if not top:
        parts = []
        for filename, content in list(kb.items())[:2]:
            parts.append(f"[Source: {filename}]\n{content[:400]}")
        return "\n\n".join(parts)
    return "\n\n".join(f"[Source: {fn}]\n{chunk}" for _, fn, chunk in top)

# ── Agent State ───────────────────────────────────────────────────────────────
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

# ── Agent Nodes ───────────────────────────────────────────────────────────────
def make_nodes(groq_api_key: str, kb: dict):

    def node_intent_classifier(state: AgentState) -> AgentState:
        client = Groq(api_key=groq_api_key)
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": """Classify buyer message into ONE of:
GREETING, PRODUCT_INQUIRY, PRICE_INQUIRY, CERTIFICATION_QUERY,
BOOKING_REQUEST, DOCUMENT_REQUEST, COMPLAINT, SPAM, COMPLEX
Return ONLY JSON: {"intent": "INTENT", "confidence": 0.95, "reasoning": "reason"}"""},
                    {"role": "user", "content": f"Classify: {state['message']}"}
                ],
                max_tokens=100, timeout=30,
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
            state["intent"] = result["intent"]
            state["intent_confidence"] = result["confidence"]
        except Exception:
            state["intent"] = "COMPLEX"
            state["intent_confidence"] = 0.5
        return state

    def node_rag_retriever(state: AgentState) -> AgentState:
        state["rag_context"] = retrieve_context(state["message"], kb)
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
                    {"role": "system", "content": """Extract buyer info from conversation.
Return ONLY JSON:
{"buyer_name":"name or unknown","company":"company or unknown",
"country":"country or unknown","product_interest":"product or unknown",
"quantity":"quantity or unknown","certification":"cert or none",
"timeline":"timeline or unknown","budget":"budget or unknown",
"qualification_complete": true/false}"""},
                    {"role": "user", "content": f"Extract from:\n{history_text}"}
                ],
                max_tokens=250, timeout=30,
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            buyer_info = json.loads(text)
            state["buyer_info"] = buyer_info

            score = 0
            numbers = re.findall(r'\d+', str(buyer_info.get("quantity", "")))
            if numbers:
                qty = int(numbers[0])
                if qty >= 5000: score += 30
                elif qty >= 1000: score += 15
                elif qty > 0: score += 5

            country = str(buyer_info.get("country", "")).lower()
            premium = ["germany","uk","united kingdom","united states","usa",
                       "france","italy","netherlands","sweden","denmark",
                       "belgium","canada","australia","japan"]
            if any(m in country for m in premium): score += 20
            elif country and country != "unknown": score += 10

            cert = str(buyer_info.get("certification", "")).lower()
            if cert and cert not in ["none","unknown",""]: score += 15

            timeline = str(buyer_info.get("timeline", "")).lower()
            if timeline and timeline not in ["unknown",""]: score += 15

            budget = str(buyer_info.get("budget", "")).lower()
            if budget and budget not in ["unknown",""]: score += 20

            state["lead_score"] = score
            state["lead_status"] = "HOT" if score >= 80 else "WARM" if score >= 50 else "COLD"
        except Exception:
            state["buyer_info"] = {}
            state["lead_score"] = 0
            state["lead_status"] = "COLD"
        return state

    def node_response_generator(state: AgentState) -> AgentState:
        client = Groq(api_key=groq_api_key)
        if state["lead_status"] == "HOT":
            closing = "This is a HOT lead. Offer to book a discovery call: https://calendly.com/textilebot/discovery"
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
                    {"role": "system", "content": f"""You are TextileBot for a Pakistani textile export company.
CONTEXT:
{state["rag_context"]}
RULES:
- Answer only from context
- Be professional and concise
- {closing}"""},
                    {"role": "user", "content": f"Conversation:\n{history_text}\n\nLatest: {state['message']}\n\nRespond:"}
                ],
                max_tokens=400, timeout=30,
            )
            state["response"] = response.choices[0].message.content.strip()
        except Exception:
            state["response"] = "Thank you for your message. Our team will respond shortly."
        return state

    def node_complaint_handler(state: AgentState) -> AgentState:
        state["escalated"] = True
        state["response"] = (
            "I sincerely apologize for the inconvenience. "
            "I am escalating your concern to our senior team immediately. "
            "A team member will contact you within 2 hours."
        )
        return state

    def node_spam_filter(state: AgentState) -> AgentState:
        state["response"] = ""
        return state

    def route_after_intent(state: AgentState) -> str:
        if state["intent"] == "COMPLAINT": return "complaint_handler"
        elif state["intent"] == "SPAM": return "spam_filter"
        else: return "rag_retriever"

    return (node_intent_classifier, node_rag_retriever, node_lead_qualifier,
            node_response_generator, node_complaint_handler, node_spam_filter,
            route_after_intent)

@st.cache_resource
def build_agent():
    groq_api_key = os.environ.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") or ""
    kb = load_knowledge_base()
    (node_intent_classifier, node_rag_retriever, node_lead_qualifier,
     node_response_generator, node_complaint_handler, node_spam_filter,
     route_after_intent) = make_nodes(groq_api_key, kb)

    graph = StateGraph(AgentState)
    graph.add_node("intent_classifier", node_intent_classifier)
    graph.add_node("rag_retriever", node_rag_retriever)
    graph.add_node("lead_qualifier", node_lead_qualifier)
    graph.add_node("response_generator", node_response_generator)
    graph.add_node("complaint_handler", node_complaint_handler)
    graph.add_node("spam_filter", node_spam_filter)
    graph.set_entry_point("intent_classifier")
    graph.add_conditional_edges("intent_classifier", route_after_intent, {
        "rag_retriever": "rag_retriever",
        "complaint_handler": "complaint_handler",
        "spam_filter": "spam_filter",
    })
    graph.add_edge("rag_retriever", "lead_qualifier")
    graph.add_edge("lead_qualifier", "response_generator")
    graph.add_edge("response_generator", END)
    graph.add_edge("complaint_handler", END)
    graph.add_edge("spam_filter", END)
    return graph.compile(), groq_api_key

# ── Session State ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lead_score" not in st.session_state:
    st.session_state.lead_score = 0
if "lead_status" not in st.session_state:
    st.session_state.lead_status = "COLD"
if "intent" not in st.session_state:
    st.session_state.intent = "-"
if "call_booked" not in st.session_state:
    st.session_state.call_booked = False
if "escalated" not in st.session_state:
    st.session_state.escalated = False

# ── Load Agent ────────────────────────────────────────────────────────────────
agent, groq_api_key = build_agent()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 TextileBot — WhatsApp AI Agent Demo")
st.caption("B2B Lead Qualification Agent for Pakistani Textile Exporters")
st.divider()

col1, col2 = st.columns([2, 1])

# ── Chat Interface ────────────────────────────────────────────────────────────
with col1:
    st.subheader("💬 WhatsApp Simulation")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "buyer":
                st.markdown(f'<div class="buyer-message">🧑 {msg["content"]}</div><div class="clearfix"></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="agent-message">🤖 {msg["content"]}</div><div class="clearfix"></div>', unsafe_allow_html=True)

    st.divider()
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Your message:", placeholder="e.g. Hi, I need 8000m of GOTS certified cotton fabric for Germany")
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        st.session_state.messages.append({"role": "buyer", "content": user_input})

        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        initial_state: AgentState = {
            "message": user_input,
            "conversation_history": history,
            "intent": "",
            "intent_confidence": 0.0,
            "buyer_info": {},
            "lead_score": 0,
            "lead_status": "COLD",
            "rag_context": "",
            "response": "",
            "call_booked": False,
            "escalated": False,
            "session_id": st.session_state.session_id
        }

        with st.spinner("TextileBot thinking..."):
            final_state = agent.invoke(initial_state)

        st.session_state.lead_score = final_state["lead_score"]
        st.session_state.lead_status = final_state["lead_status"]
        st.session_state.intent = final_state["intent"]
        st.session_state.call_booked = final_state["call_booked"]
        st.session_state.escalated = final_state["escalated"]

        if final_state["response"]:
            st.session_state.messages.append({"role": "agent", "content": final_state["response"]})

        st.rerun()

# ── Lead Dashboard ────────────────────────────────────────────────────────────
with col2:
    st.subheader("📊 Live Lead Dashboard")

    status_color = {"HOT": "🔴", "WARM": "🟡", "COLD": "🔵"}.get(st.session_state.lead_status, "⚪")
    st.metric("Lead Score", f"{st.session_state.lead_score}/100")
    st.metric("Lead Status", f"{status_color} {st.session_state.lead_status}")
    st.metric("Intent", st.session_state.intent)

    st.divider()

    if st.session_state.call_booked:
        st.success("✅ Discovery call offered!")
    if st.session_state.escalated:
        st.warning("⚠️ Escalated to human agent!")
    else:
        st.info("🤖 Handled by AI")

    st.divider()
    st.write("**Session Info:**")
    st.caption(f"Session ID: {st.session_state.session_id}")
    st.caption(f"Messages: {len(st.session_state.messages)}")

    st.divider()
    st.write("**Score Criteria:**")
    st.caption("📦 Quantity ≥5000m: +30")
    st.caption("🌍 Premium market: +20")
    st.caption("📜 Certification required: +15")
    st.caption("📅 Timeline specified: +15")
    st.caption("💰 Budget mentioned: +20")

    st.divider()
    st.write("**Built with:**")
    st.caption("⚡ LangGraph — Agent framework")
    st.caption("🔍 Keyword Search — RAG retrieval")
    st.caption("🤖 Groq — LLM inference")
    st.caption("🎨 Streamlit — This UI")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.lead_score = 0
        st.session_state.lead_status = "COLD"
        st.session_state.intent = "-"
        st.session_state.call_booked = False
        st.session_state.escalated = False
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()