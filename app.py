import io
import contextlib
import time

import streamlit as st

from crew_agent import build_crew, log_interaction, log_feedback


st.set_page_config(
    page_title="Anandakrishna Residency AI Assistant",
    page_icon="🏨",
    layout="centered"
)


# -----------------------------
# Session State Initialization
# -----------------------------

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "adaptive_mode" not in st.session_state:
    st.session_state.adaptive_mode = False

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("⚙️ Assistant Settings")

debug_mode = st.sidebar.toggle(
    "Show agent/tool execution details",
    value=False
)

st.sidebar.markdown("---")
st.sidebar.write("**Memory turns stored:**", len(st.session_state.conversation_history))

if st.sidebar.button("Reset Memory & Adaptive Mode"):
    st.session_state.conversation_history = []
    st.session_state.messages = []
    st.session_state.adaptive_mode = False
    st.session_state.last_question = None
    st.session_state.last_answer = None
    st.rerun()


# -----------------------------
# Header
# -----------------------------

st.title("🏨 Anandakrishna Residency AI Assistant")
st.caption("Ask about room tariffs, facilities, Guruvayoor Temple, availability, booking, or cancellation.")


# -----------------------------
# Display Chat History
# -----------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# -----------------------------
# Chat Input
# -----------------------------

user_query = st.chat_input("Type your question here...")

if user_query:
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.write(user_query)

    if user_query.lower() in ["reset", "clear memory", "new conversation"]:
        st.session_state.conversation_history = []
        st.session_state.adaptive_mode = False
        answer = "Conversation memory and adaptive mode have been cleared."
    else:
        start_time = time.time()

        try:
            crew = build_crew(
                user_query,
                debug_mode,
                st.session_state.conversation_history,
                st.session_state.adaptive_mode
            )

            debug_buffer = io.StringIO()

            with contextlib.redirect_stdout(debug_buffer):
                result = crew.kickoff()

            debug_logs = debug_buffer.getvalue()
            answer = str(result)

            if debug_mode and debug_logs:
                with st.expander("🔍 How the AI answered (Tool Trace)"):
                    st.code(debug_logs)

            latency = round(time.time() - start_time, 2)
            st.caption(f"⏱ Response time: {latency} sec")

            try:
                log_interaction(user_query, answer)
            except Exception as log_error:
                if debug_mode:
                    st.warning(f"Interaction log could not be saved: {log_error}")

            st.session_state.conversation_history.append({
                "user": user_query,
                "agent": answer
            })

            st.session_state.conversation_history = st.session_state.conversation_history[-5:]

        except Exception as error:
            answer = (
                "Sorry, I am unable to respond right now due to a technical issue. "
                "Please contact the front office for assistance."
            )
            if debug_mode:
                st.error(f"Technical error: {error}")
            else:
                st.warning("⚠️ Something went wrong. Please try again later.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.session_state.last_question = user_query
    st.session_state.last_answer = answer

    with st.chat_message("assistant"):
        st.write(answer)


# -----------------------------
# Feedback Section
# -----------------------------

if st.session_state.last_question and st.session_state.last_answer:
    st.markdown("---")
    st.subheader("Feedback")
    st.caption("Your feedback helps improve future responses during this session.")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("👍 Helpful"):
            try:
                log_feedback(
                    st.session_state.last_question,
                    st.session_state.last_answer,
                    "yes"
                )
                st.success("Thank you for the feedback.")
            except Exception as feedback_error:
                if debug_mode:
                    st.error(f"Feedback could not be saved: {feedback_error}")
                else:
                    st.warning("⚠️ Unable to save feedback right now.")

    with col2:
        if st.button("👎 Not Helpful"):
            try:
                log_feedback(
                    st.session_state.last_question,
                    st.session_state.last_answer,
                    "no"
                )
                st.session_state.adaptive_mode = True
                st.warning("Feedback noted. Future responses will be more cautious and explanatory.")
            except Exception as feedback_error:
                if debug_mode:
                    st.error(f"Feedback could not be saved: {feedback_error}")
                else:
                    st.warning("⚠️ Unable to save feedback right now.")

    with col3:
        if st.button("Skip"):
            st.info("Feedback skipped.")
