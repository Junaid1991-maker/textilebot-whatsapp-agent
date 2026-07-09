# TextileBot — WhatsApp AI Agent for B2B Textile Businesses

A production-grade AI agent that automatically handles WhatsApp inquiries for Pakistani textile exporters. Built with LangGraph, RAG, ChromaDB, FastAPI, and Streamlit.

## What It Does

A buyer messages you on WhatsApp at 2am. You are sleeping. TextileBot:
- Replies instantly and professionally
- Qualifies the lead with smart questions
- Scores them 0-100 automatically
- Books a discovery call if HOT lead
- Escalates complaints to humans
- Logs everything to Google Sheets

## Tech Stack

- **LangGraph** — Multi-node agent framework
- **ChromaDB** — Vector database for RAG
- **Groq** — LLM inference (llama-3.1-8b-instant)
- **FastAPI** — Production REST API
- **Streamlit** — Demo frontend UI
- **Python 3.11** — Core language

## Architecture

WhatsApp Message
↓
Intent Classifier (9 categories)
↓
Router → RAG Retriever → Lead Qualifier → Lead Scorer → Response Generator
↓                                                        ↓
Complaint Handler                                    Calendly Booking
↓
Spam Filter

## Quick Start

1. Clone the repository
2. Create virtual environment: `python -m venv textilebot_env`
3. Activate: `textilebot_env\Scripts\activate`
4. Install: `pip install -r requirements.txt`
5. Create `.env` file from `.env.example`
6. Start API: `python src/api/main.py`
7. Start UI: `streamlit run streamlit_app/app.py`

## Project Structure

textilebot-whatsapp-agent/
├── data/
│   ├── synthetic/          # 50 buyer profiles, 20 conversations
│   └── knowledge_base/     # 7 domain knowledge documents
├── src/
│   ├── api/main.py         # FastAPI backend
│   ├── agent/              # LangGraph agent nodes
│   └── utils/              # Helper utilities
├── streamlit_app/app.py    # Demo UI
├── notebooks/              # Step by step Jupyter demos
└── requirements.txt

## Key Features

- **RAG Pipeline** — Answers grounded in real business documents
- **Lead Scoring** — Automatic 0-100 scoring with breakdown
- **Security** — Input validation, rate limiting, prompt injection protection
- **Multi-turn** — Score updates with every message
- **Free to run** — Groq free tier, no GPU needed

## Agent Nodes

| Node | Function |
|------|----------|
| Intent Classifier | 9 categories — GREETING to SPAM |
| RAG Retriever | Finds relevant knowledge base chunks |
| Lead Qualifier | Extracts buyer info from conversation |
| Lead Scorer | Scores 0-100 based on 5 criteria |
| Response Generator | Grounded answers from documents |
| Complaint Handler | Empathy + human escalation |
| Spam Filter | Silent rejection |

## Built By

**M. Junaid Iqbal** — Agentic AI Engineer
- Website: iamjunaidiqbal.com
- LinkedIn: linkedin.com/in/junaid-iqbal
- GitHub: github.com/Junaid1991-maker

