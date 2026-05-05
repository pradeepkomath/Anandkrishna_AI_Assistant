import os
import csv
from pathlib import Path
from datetime import datetime
import contextlib
import io

import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

from tools import (
    rag_knowledge_tool,
    availability_check_tool,
    human_escalation_tool,
    room_tariff_lookup_tool
)


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LOG_FILE = BASE_DIR / "crew_agent_logs.csv"
FEEDBACK_LOG_FILE = BASE_DIR / "feedback_logs.csv"


def get_openai_api_key() -> str:
    """Read API key from Streamlit Cloud secrets first, then fall back to local .env."""
    try:
        key_from_secrets = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key_from_secrets = None

    return key_from_secrets or os.getenv("OPENAI_API_KEY", "")


def log_interaction(question: str, answer: str) -> None:
    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["timestamp", "question", "answer", "model"])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            question,
            answer,
            OPENAI_MODEL
        ])


def log_feedback(question: str, answer: str, feedback: str) -> None:
    file_exists = FEEDBACK_LOG_FILE.exists()

    with open(FEEDBACK_LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["timestamp", "question", "answer", "feedback"])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            question,
            answer,
            feedback
        ])


def build_crew(customer_query: str, debug_mode: bool, conversation_history: list, adaptive_mode: bool) -> Crew:
    api_key = get_openai_api_key()

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Add it locally in .env or in Streamlit Cloud Secrets.")

    llm = LLM(
        model=f"openai/{OPENAI_MODEL}",
        temperature=0.2,
        api_key=api_key
    )

    adaptive_instruction = ""

    if adaptive_mode:
        adaptive_instruction = """
Adaptive behaviour instruction:
User feedback indicates dissatisfaction.
Respond more cautiously and explicitly indicate uncertainty.
Always start the response with: "Based on previous feedback,"
"""

    formatted_history = ""
    for turn in conversation_history[-5:]:
        formatted_history += f"Customer: {turn['user']}\nAgent: {turn['agent']}\n\n"

    hotel_support_agent = Agent(
        role="Anandakrishna Residency Customer Support Agent",
        goal=(
            "Answer customer questions safely using the correct tool. "
            "Use RAG for information, availability tool for date-based availability checks, "
            "room tariff tool for pricing questions, and escalation tool for business-impacting requests."
        ),
        backstory=(
            "You are a virtual customer support assistant for Anandakrishna Residency, "
            "a lodging facility in Guruvayoor. You help pilgrims and guests with room, "
            "facility, tariff, and Guruvayoor Temple related questions. You must avoid "
            "unsafe commitments such as confirming bookings, cancellations, negotiated quotes, "
            "or real-time availability."
        ),
        tools=[
            room_tariff_lookup_tool,
            rag_knowledge_tool,
            availability_check_tool,
            human_escalation_tool
        ],
        llm=llm,
        verbose=debug_mode,
        allow_delegation=False,
        max_iter=25
    )

    customer_support_task = Task(
        description=f"""
Handle the following customer query safely:

Previous conversation:
{formatted_history}

Customer query:
{customer_query}

Planning / reasoning rules:
For complex queries, reason through the request in steps:
1. Identify customer intent.
2. Extract key details such as pax, room type, date, or location.
3. Select the correct tool.
4. Combine tool results with previous conversation history.
5. Give a concise final answer.

{adaptive_instruction}

Tool calling rule:
When calling any tool, always pass the customer query as a plain string using the parameter named description.
Example: {{"description": "do you have room for 8 persons"}}
Do not use query, input_text, Any objects, schema objects, or nested dictionaries.

Tool usage rules:
1. Use room_tariff_lookup_tool for:
   - room tariff
   - room rate
   - room pricing
   - pax capacity
   - suitable room recommendation
   - room category questions
   - room for 8 persons
   - room for 6 pax
   - which room is suitable for 4 people
2. Use rag_knowledge_tool for informational questions about property facilities,
   Guruvayoor temple timings, poojas, festivals, and rush days.
3. Use availability_check_tool only when the customer asks about room availability AND provides a specific check-in date.
   Do not use availability_check_tool for pax capacity questions like "room for 8 persons".
   For pax/capacity/suitable room questions, use room_tariff_lookup_tool.
4. Use human_escalation_tool for booking, cancellation, modification, group quote, discount,
   negotiation, complaint, or other business-impacting requests.
5. Do not confirm bookings or cancellations.
6. Do not invent facts.
7. Keep the final response concise and polite.
""",
        expected_output=(
            "A concise customer-facing answer that either provides retrieved information, "
            "a preliminary availability result, or an escalation message."
        ),
        agent=hotel_support_agent
    )

    return Crew(
        agents=[hotel_support_agent],
        tasks=[customer_support_task],
        process=Process.sequential,
        verbose=debug_mode,
        memory=False,
        max_iter=25
    )


def main():
    if not get_openai_api_key():
        raise ValueError("OPENAI_API_KEY not found.")

    conversation_history = []

    print("Anandakrishna Residency CrewAI Assistant")
    print("Ask about room tariffs, facilities, Guruvayoor temple, availability, booking, or cancellation.")
    print("Type 'exit' to quit.\n")

    print("Select Output Mode:")
    print("1. Clean Output (Question & Answer only)")
    print("2. Debug Output (Show Agent Tool Execution Details)")
    print()

    mode = input("Enter choice (1 or 2): ").strip()
    debug_mode = mode == "2"
    adaptive_mode = False

    print()

    while True:
        customer_query = input("Customer: ").strip()

        if customer_query.lower() in ["exit", "quit"]:
            print("Agent: Thank you for contacting Anandakrishna Residency.")
            break

        if customer_query.lower() in ["reset", "clear memory", "new conversation"]:
            conversation_history.clear()
            adaptive_mode = False
            print("Agent: Conversation memory and adaptive mode have been cleared.\n")
            continue

        try:
            crew = build_crew(customer_query, debug_mode, conversation_history, adaptive_mode)

            if debug_mode:
                result = crew.kickoff()
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = crew.kickoff()

            answer = str(result)

            log_interaction(customer_query, answer)
            conversation_history.append({
                "user": customer_query,
                "agent": answer
            })
            conversation_history = conversation_history[-5:]

            print(f"\nAgent: {answer}\n")

            feedback = input("Was this response helpful? (yes/no/skip): ").strip().lower()

            if feedback in ["yes", "no"]:
                log_feedback(customer_query, answer, feedback)

            if feedback == "no":
                adaptive_mode = True
                print("Agent: Feedback noted. I will provide more cautious and detailed responses in future.\n")
            elif feedback == "yes":
                print("Agent: Thank you for the feedback.\n")

        except Exception as error:
            print(
                "\nAgent: Sorry, I am unable to respond right now due to a technical issue. "
                "Please contact the front office for assistance.\n"
            )
            print(f"Technical error: {error}\n")


if __name__ == "__main__":
    main()
