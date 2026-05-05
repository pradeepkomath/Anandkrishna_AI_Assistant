import os
import re
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from crewai.tools import tool

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

FAISS_INDEX_PATH = (BASE_DIR / os.getenv("FAISS_INDEX_PATH", "vectorstore/faiss_index")).resolve()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("TOP_K", "4"))

FRONT_OFFICE_CONTACT = "0487 2555869 / 9567566777"

ROOM_TARIFFS = {
    "premium": {"pax": 8, "plan": "A/C", "rate": 5775},
    "suite": {"pax": 6, "plan": "A/C", "rate": 5250},
    "4 bed deluxe": {"pax": 4, "plan": "A/C", "rate": 2888},
    "4 bed": {"pax": 4, "plan": "A/C", "rate": 2625},
    "2 bed": {"pax": 2, "plan": "A/C", "rate": 1838},
}


RAG_SYSTEM_PROMPT = """
You are a customer support assistant for Anandakrishna Residency, Guruvayoor.

Answer ONLY using the retrieved context.

Rules:
- Do not invent facts.
- Do not confirm bookings, cancellations, availability, or negotiated quotations.
- If information is missing, say it is not available in the current knowledge base.
- Keep the answer polite and concise.

Retrieved context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "{question}")
])


def _load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if not FAISS_INDEX_PATH.exists():
        return None

    return FAISS.load_local(
        str(FAISS_INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )


@tool("rag_knowledge_tool")
def rag_knowledge_tool(description: Any = None, query: Any = None, input_text: Any = None) -> str:
    """
    Use this tool for informational questions about:
    - hotel facilities
    - Guruvayoor Temple timings
    - poojas, festivals, and rush days

    Do not use this tool for booking confirmation, cancellation, negotiation, or real-time availability.
    """
    raw_input = description or query or input_text or ""

    if isinstance(raw_input, dict):
        query = (
            raw_input.get("description")
            or raw_input.get("query")
            or raw_input.get("input_text")
            or ""
        ).lower()
    else:
        query = str(raw_input).lower()

    
    vectorstore = _load_vectorstore()

    if vectorstore is None:
        return "Knowledge base is not available. Please run ingest.py before using this tool."

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(query)

    if not docs:
        return (
            "The information is not available in the current knowledge base. "
            f"Please contact the front office at {FRONT_OFFICE_CONTACT}."
        )

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    messages = rag_prompt.format_messages(context=context, question=query)
    response = llm.invoke(messages)

    return response.content


@tool("availability_check_tool")
def availability_check_tool(description: Any = None, query: Any = None, input_text: Any = None) -> str:
    """
    Use this tool only when the customer asks about room availability for a specific check-in date.
    This is a mock availability checker for demonstration.
    Odd date = rooms may be available.
    Even date = rooms may be unavailable.
    It must not confirm booking.
    """
    raw_input = description or query or input_text or ""
    if isinstance(raw_input, dict):
        query = (
            raw_input.get("description")
            or raw_input.get("query")
            or raw_input.get("input_text")
            or ""
        )
    else:
        query = str(raw_input)
    day = _extract_day_from_query(query)

    if day is None:
        return (
            "I need a check-in date to perform a preliminary availability check. "
            f"Please contact the front office at {FRONT_OFFICE_CONTACT} for exact availability."
        )

    if day % 2 == 1:
        return (
            f"Preliminary availability check for day {day}: rooms may be available. "
            "This is not a confirmed reservation. Please contact the front office for confirmation."
        )

    return (
        f"Preliminary availability check for day {day}: rooms may be limited or unavailable. "
        "Please contact the front office for exact confirmation."
    )


@tool("human_escalation_tool")
def human_escalation_tool(description: Any = None, query: Any = None, input_text: Any = None) -> str:
    """
    Use this tool for booking, cancellation, modification, group quotation, discount,
    negotiation, complaint, or any business-impacting request.
    """
    raw_input = description or query or input_text or ""
    if isinstance(raw_input, dict):
        query = raw_input.get("description", "")   
    else:
        query = str(raw_input)

    # Query extraction is intentionally not used in the current version.
    # This is kept for future enhancements such as intent-specific escalation handling.
        
    return (
        "This request requires human staff assistance. "
        f"Please contact Anandakrishna Residency front office at {FRONT_OFFICE_CONTACT}. "
        "The AI assistant cannot confirm bookings, cancel reservations, modify bookings, "
        "or provide negotiated quotations."
    )




@tool("room_tariff_lookup_tool")
def room_tariff_lookup_tool(description: Any = None, query: Any = None, input_text: Any = None) -> str:
    """
    Use this tool for room tariff, rate, price, rent, or pax capacity questions.
    This tool gives structured tariff information for Anandakrishna Residency.
    """
    raw_input = description or query or input_text or ""
    if isinstance(raw_input, dict):
        text = (
            raw_input.get("description")
            or raw_input.get("query")
            or raw_input.get("input_text")
            or ""
        ).lower()
    else:
        text = str(raw_input).lower()

   

    if "premium" in text or "8 bed" in text or "octaple" in text or "8 pax" in text or "8" in text or "8 person" in text:
        room_type = "premium"
    elif "suite" in text or "suit" in text or "6 bed" in text or "sextuple" in text or "6" in text:
        room_type = "suite"
    elif "4 bed deluxe" in text or "four bed deluxe" in text or "4" in text:
        room_type = "4 bed deluxe"
    elif "4 bed" in text or "four bed" in text or "4 pax" in text:
        room_type = "4 bed"
    elif "2 bed" in text or "two bed" in text or "double" in text or "2 pax" in text or "2" in text:
        room_type = "2 bed"
    else:
        available = ", ".join(ROOM_TARIFFS.keys())
        return (
            "Please specify the room type. Available room types are: "
            f"{available}."
        )

    details = ROOM_TARIFFS[room_type]

    return (
        f"The tariff for {room_type.upper()} room is ₹{details['rate']} per room. "
        f"It is an {details['plan']} room with pax capacity of {details['pax']}. "
        "Tariffs are subject to confirmation by the front office."
    )

def _extract_day_from_query(query: str):
    text = query.lower()

    # Matches dates like 01-06-2026, 1/6/2026, 1.6.2026
    numeric_date = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})?\b", text)
    if numeric_date:
        return int(numeric_date.group(1))

    # Matches "June 1", "June 1st", "1 June", "1st June"
    month_names = (
        "january|february|march|april|may|june|july|august|"
        "september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
    )

    month_first = re.search(rf"\b({month_names})\s+(\d{{1,2}})(st|nd|rd|th)?\b", text)
    if month_first:
        return int(month_first.group(2))

    day_first = re.search(rf"\b(\d{{1,2}})(st|nd|rd|th)?\s+({month_names})\b", text)
    if day_first:
        return int(day_first.group(1))

    # Matches plain "on 1" or "date 1"
    simple_day = re.search(r"\b(?:on|date|day)\s+(\d{1,2})\b", text)
    if simple_day:
        return int(simple_day.group(1))

    return None