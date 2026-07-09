# streamlit_app/app.py
# TextileBot — WhatsApp AI Agent Demo
# This is the visual frontend that shows TextileBot in action
# Left side: chat interface like WhatsApp
# Right side: live lead dashboard

import streamlit as st
import requests
import json
from datetime import datetime

# ── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="TextileBot — WhatsApp AI Agent Demo",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────────────────────────
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
        border: 1px solid #e0e0e0;
    }
    .timestamp {
        font-size: 11px;
        color: #999;
        margin: 2px 5px;
    }
    .hot-badge {
        background-color: #ff4444;
        color: white;
        padding: 3px 10px;
        border-radius: 10px;
        font-weight: bold;
    }
    .warm-badge {
        background-color: #ff8800;
        color: white;
        padding: 3px 10px;
        border-radius: 10px;
        font-weight: bold;
    }
    .cold-badge {
        background-color: #4444ff;
        color: white;
        padding: 3px 10px;
        border-radius: 10px;
        font-weight: bold;
    }
    .chat-container {
        background-color: #e5ddd5;
        padding: 20px;
        border-radius: 10px;
        min-height: 400px;
        max-height: 500px;
        overflow-y: auto;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────
import os
API_URL = os.environ.get("API_URL", "http://localhost:8000")

SCENARIO_MESSAGES = {
    "HOT Lead — German Buyer": [
        "Hello, I am Klaus from Berlin Textiles GmbH in Germany",
        "We need OEKO-TEX certified cotton fabric, 10000 meters per month",
        "Our budget is $7-10 per meter CIF Hamburg, delivery in 45 days",
        "Can we book a call to discuss further?"
    ],
    "WARM Lead — UK Retailer": [
        "Hi there, I found you on LinkedIn",
        "We are a UK retailer interested in home textiles",
        "What certifications do you have?",
        "What is your MOQ for bed linen?"
    ],
    "COLD Lead — Student Research": [
        "Hello, I am a student doing research on textile exports",
        "Can you tell me generally how textile exports work?",
        "Thank you for the information"
    ],
    "Complaint — Late Shipment": [
        "My order from 3 weeks ago still has not arrived, this is completely unacceptable!"
    ],
    "Certification Query": [
        "Do you have GOTS certification for organic cotton?",
        "We specifically need organic certified fabric for our brand"
    ],
    "Booking Request": [
        "Hello, I want to book a discovery call to discuss a large order",
        "We need 50000 meters of denim fabric annually from Germany"
    ],
    "Spam Message": [
        "MAKE $10000 PER DAY WITH CRYPTO!!! Click here now!!!"
    ],
}


# ── Helper Functions ─────────────────────────────────────────────

def check_api_health() -> bool:
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def send_message(message: str, session_id: str, history: list) -> dict:
    """
    Send a message to the FastAPI backend and get response.
    
    Args:
        message: Buyer message to send.
        session_id: Unique session identifier.
        history: Conversation history so far.
        
    Returns:
        API response dictionary or error dict.
    """
    try:
        payload = {
            "session_id": session_id,
            "buyer_message": message,
            "conversation_history": history
        }
        response = requests.post(
            f"{API_URL}/respond",
            json=payload,
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Make sure the FastAPI server is running."}
    except Exception as e:
        return {"error": str(e)}


def get_status_badge(status: str) -> str:
    """Return HTML badge for lead status."""
    if status == "HOT":
        return '<span class="hot-badge">🔥 HOT</span>'
    elif status == "WARM":
        return '<span class="warm-badge">⚡ WARM</span>'
    else:
        return '<span class="cold-badge">❄️ COLD</span>'


def render_score_meter(score: int) -> None:
    """Render a visual score meter using Streamlit progress bar."""
    if score >= 80:
        color = "🔴"
    elif score >= 50:
        color = "🟡"
    else:
        color = "🔵"
    
    st.metric(
        label="Lead Score",
        value=f"{score}/100",
        delta=None
    )
    st.progress(score / 100)
    st.write(f"{color} Score: {score}/100")


# ── Initialize Session State ─────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "lead_score" not in st.session_state:
    st.session_state.lead_score = 0

if "lead_status" not in st.session_state:
    st.session_state.lead_status = "COLD"

if "last_intent" not in st.session_state:
    st.session_state.last_intent = "—"

if "call_booked" not in st.session_state:
    st.session_state.call_booked = False

if "escalated" not in st.session_state:
    st.session_state.escalated = False

if "scenario_index" not in st.session_state:
    st.session_state.scenario_index = 0


# ── Main Layout ──────────────────────────────────────────────────

st.title("🧵 TextileBot — WhatsApp AI Agent Demo")
st.caption("B2B Textile Export Business — Powered by LangGraph + RAG + Groq AI")

# API Status check
api_ok = check_api_health()
if api_ok:
    st.success("✅ API Connected — TextileBot is ready")
else:
    st.error("❌ API Offline — Start the FastAPI server first: python src/api/main.py")

st.divider()

# Two column layout
col_chat, col_dashboard = st.columns([3, 2])


# ── LEFT COLUMN — Chat Interface ─────────────────────────────────

with col_chat:
    st.subheader("💬 WhatsApp Chat Simulation")
    
    # Scenario buttons
    st.write("**Quick Scenarios — Click to auto-fill:**")
    
    scenario_cols = st.columns(3)
    scenarios = list(SCENARIO_MESSAGES.keys())
    
    selected_scenario = None
    for i, scenario in enumerate(scenarios[:6]):
        col_idx = i % 3
        with scenario_cols[col_idx]:
            if st.button(scenario.split("—")[0].strip(), key=f"btn_{i}"):
                selected_scenario = scenario
    
    # Handle scenario selection
    if selected_scenario:
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.lead_score = 0
        st.session_state.lead_status = "COLD"
        st.session_state.call_booked = False
        st.session_state.escalated = False
        st.session_state.last_intent = "—"
        import uuid
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.pending_scenario = selected_scenario
        st.session_state.scenario_index = 0
    
    # Send next scenario message button
    if "pending_scenario" in st.session_state:
        scenario_msgs = SCENARIO_MESSAGES[st.session_state.pending_scenario]
        idx = st.session_state.scenario_index
        
        if idx < len(scenario_msgs):
            next_msg = scenario_msgs[idx]
            st.info(f"Next message: *{next_msg}*")
            
            if st.button("▶ Send This Message", key="send_scenario"):
                with st.spinner("TextileBot is typing..."):
                    result = send_message(
                        message=next_msg,
                        session_id=st.session_state.session_id,
                        history=st.session_state.conversation_history
                    )
                
                if "error" not in result:
                    # Add to display messages
                    st.session_state.messages.append({
                        "role": "buyer",
                        "content": next_msg,
                        "time": datetime.now().strftime("%H:%M")
                    })
                    
                    if result.get("response"):
                        st.session_state.messages.append({
                            "role": "agent",
                            "content": result["response"],
                            "time": datetime.now().strftime("%H:%M")
                        })
                    
                    # Update conversation history
                    st.session_state.conversation_history.append({
                        "role": "buyer",
                        "content": next_msg
                    })
                    if result.get("response"):
                        st.session_state.conversation_history.append({
                            "role": "agent",
                            "content": result["response"]
                        })
                    
                    # Update dashboard data
                    st.session_state.lead_score = result.get("lead_score", 0)
                    st.session_state.lead_status = result.get("lead_status", "COLD")
                    st.session_state.last_intent = result.get("intent", "—")
                    st.session_state.call_booked = result.get("call_booked", False)
                    st.session_state.escalated = result.get("escalated", False)
                    st.session_state.scenario_index += 1
                    
                    st.rerun()
                else:
                    st.error(result["error"])
        else:
            st.success("✅ Scenario complete")
            if st.button("Clear and Start New"):
                del st.session_state.pending_scenario
                st.session_state.scenario_index = 0
                st.rerun()
    
    # Manual message input
    st.write("**Or type your own message:**")
    with st.form("manual_message", clear_on_submit=True):
        user_input = st.text_input(
            "Type a message",
            placeholder="e.g. I need 5000 meters of cotton fabric from Germany"
        )
        send_button = st.form_submit_button("Send 📤")
    
    if send_button and user_input:
        with st.spinner("TextileBot is typing..."):
            result = send_message(
                message=user_input,
                session_id=st.session_state.session_id,
                history=st.session_state.conversation_history
            )
        
        if "error" not in result:
            st.session_state.messages.append({
                "role": "buyer",
                "content": user_input,
                "time": datetime.now().strftime("%H:%M")
            })
            if result.get("response"):
                st.session_state.messages.append({
                    "role": "agent",
                    "content": result["response"],
                    "time": datetime.now().strftime("%H:%M")
                })
            
            st.session_state.conversation_history.append({
                "role": "buyer", "content": user_input
            })
            if result.get("response"):
                st.session_state.conversation_history.append({
                    "role": "agent", "content": result["response"]
                })
            
            st.session_state.lead_score = result.get("lead_score", 0)
            st.session_state.lead_status = result.get("lead_status", "COLD")
            st.session_state.last_intent = result.get("intent", "—")
            st.session_state.call_booked = result.get("call_booked", False)
            st.session_state.escalated = result.get("escalated", False)
            st.rerun()
        else:
            st.error(result["error"])
    
    # Chat display
    st.write("**Conversation:**")
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            st.caption("No messages yet. Click a scenario button or type a message above.")
        
        for msg in st.session_state.messages:
            if msg["role"] == "buyer":
                st.markdown(
                    f'<div class="buyer-message">👤 <b>Buyer:</b> {msg["content"]}'
                    f'<div class="timestamp">{msg["time"]}</div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="agent-message">🤖 <b>TextileBot:</b> {msg["content"]}'
                    f'<div class="timestamp">{msg["time"]}</div></div>',
                    unsafe_allow_html=True
                )
            st.write("")
    
    # Clear button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.lead_score = 0
        st.session_state.lead_status = "COLD"
        st.session_state.call_booked = False
        st.session_state.escalated = False
        st.session_state.last_intent = "—"
        if "pending_scenario" in st.session_state:
            del st.session_state.pending_scenario
        st.rerun()


# ── RIGHT COLUMN — Live Dashboard ────────────────────────────────

with col_dashboard:
    st.subheader("📊 Live Lead Dashboard")
    
    # Lead Score
    st.write("**Lead Score:**")
    render_score_meter(st.session_state.lead_score)
    
    st.divider()
    
    # Lead Status Badge
    st.write("**Lead Status:**")
    st.markdown(
        get_status_badge(st.session_state.lead_status),
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Last Intent
    st.write("**Last Detected Intent:**")
    st.code(st.session_state.last_intent)
    
    st.divider()
    
    # Actions taken
    st.write("**Actions Taken:**")
    
    if st.session_state.call_booked:
        st.success("📅 Calendly link sent — Call booked!")
    else:
        st.info("📅 No call booked yet")
    
    if st.session_state.escalated:
        st.warning("🚨 Escalated to human agent!")
    else:
        st.info("🤖 Handled by AI")
    
    st.divider()
    
    # Session info
    st.write("**Session Info:**")
    st.caption(f"Session ID: {st.session_state.session_id}")
    st.caption(f"Messages: {len(st.session_state.messages)}")
    
    st.divider()
    
    # Score explanation
    st.write("**Score Criteria:**")
    st.caption("✅ Quantity ≥ 5000m: +30")
    st.caption("✅ Premium market: +20")
    st.caption("✅ Certification required: +15")
    st.caption("✅ Timeline specified: +15")
    st.caption("✅ Budget mentioned: +20")
    
    st.divider()
    
    st.write("**Built with:**")
    st.caption("🔗 LangGraph — Agent framework")
    st.caption("🔗 ChromaDB — Vector database")
    st.caption("🔗 Groq — LLM inference")
    st.caption("🔗 FastAPI — Backend API")
    st.caption("🔗 Streamlit — This UI")