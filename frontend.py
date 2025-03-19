import os
import streamlit as st
import requests
import uuid
from streamlit_modal import Modal

# Custom icons for user and assistant
user_icon = "https://cdn.vectorstock.com/i/1000v/74/41/white-user-icon-vector-42797441.jpg"
# assistant_icon = "https://cdn.prod.website-files.com/677e635f0494590a26a937e9/677e66b78bd3512252bbc315_Logo.png"
assistant_icon = "halsa_logo.png"

GENERAL_QUESTIONS = [
    "Summarize my health data",
    "What actions can I take to improve my health?",
]

SPECIFIC_QUESTIONS = [
    "I feel stressed. What should I do?",
    "Is my blood pressure within normal range?",
    "Are there any risk factors for cardiovascular disease or diabetes?",
    "How can I improve my sleep quality?",
]

BACKEND_QUERY_URL = "http://localhost:9009/query/"
BACKEND_FEEDBACK_URL = "http://localhost:9009/feedback/"

def main():
    initialise_chat()
    display_suggestions()
    display_chat_history()
    handle_user_input()
    # If feedback has been triggered, show the modal.
    if st.session_state.feedback_modal_index is not None:
        show_feedback_modal(st.session_state.feedback_modal_index)

def initialise_chat():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "suggestions_visible" not in st.session_state:
        st.session_state.suggestions_visible = True
    if "trace_id" not in st.session_state:
        st.session_state.trace_id = None  # Will be set from /query/ response
    if "feedback_modal_index" not in st.session_state:
        st.session_state.feedback_modal_index = None

def display_suggestions():
    if st.session_state.suggestions_visible:
        st.markdown("### Need some ideas? Try these:")
        st.markdown("###### General Questions")
        display_suggestion_buttons(GENERAL_QUESTIONS, key_prefix="general")
        st.markdown("###### More Specific Questions")
        display_suggestion_buttons(SPECIFIC_QUESTIONS, key_prefix="specific")

def display_suggestion_buttons(questions, key_prefix):
    for i, question in enumerate(questions):
        button_key = f"{key_prefix}_question_{i}"
        if st.button(question, key=button_key):
            add_user_input(question)
            st.session_state.suggestions_visible = False
            st.rerun()

def display_chat_history():
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user", avatar=user_icon):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar=assistant_icon):
                st.markdown(message["content"])
                # Display the thumbs widget to capture feedback
                st.feedback("thumbs", key=f"feedback_{i}", on_change=on_feedback_change, args=[i])

def on_feedback_change(index):
    """
    When the user clicks the thumbs widget, instead of immediately sending feedback,
    store the index of the message that needs feedback in session_state and trigger a rerun.
    """
    st.session_state.feedback_modal_index = index
    st.rerun()

def show_feedback_modal(index):
    """
    Uses streamlit_modal to open a modal dialog for optional comment.
    The modal collects an optional comment and submits the complete feedback.
    """
    # Create a modal instance; each message gets its own key for isolation.
    modal = Modal(key=f"feedback_modal_{index}", title="Additional Feedback")
    # Open the modal if it isn't already open.
    if not modal.is_open():
        modal.open()

    if modal.is_open():
        comment = st.text_area("Please add your comment (optional):", key=f"modal_comment_{index}")
        if st.button("Submit Feedback", key=f"submit_feedback_{index}"):
            submit_feedback_modal(index, comment)
            # Clear the modal flag and close the modal
            st.session_state.feedback_modal_index = None
            modal.close()
            st.rerun()

def submit_feedback_modal(index, comment):
    """
    Gather the thumbs value, the optional comment, and other related details,
    then send the feedback to the backend.
    """
    feedback_value = st.session_state.get(f"feedback_{index}")  # 0 or 1
    assistant_msg = st.session_state.chat_history[index]
    trace_id = st.session_state.get("trace_id")
    # Find the preceding user query
    user_query = ""
    for j in range(index - 1, -1, -1):
        if st.session_state.chat_history[j]["role"] == "user":
            user_query = st.session_state.chat_history[j]["content"]
            break

    mapped_category = "correct" if feedback_value == 1 else "incorrect"

    payload = {
        "query": user_query,
        "response": assistant_msg["content"],
        "feedback_value": mapped_category,
        "trace_id": trace_id,
        "comment": comment
    }

    try:
        resp = requests.post(BACKEND_FEEDBACK_URL, json=payload, timeout=10)
        if resp.status_code == 200:
            st.success("Thank you for your feedback!")
        else:
            st.error(f"Feedback error: {resp.status_code} - {resp.text}")
    except Exception as e:
        st.error(f"Could not submit feedback: {e}")

def handle_user_input():
    user_input = st.chat_input("Ask me anything about your health based on your data…")
    if user_input:
        add_user_input(user_input)

def add_user_input(user_input):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=user_icon):
        st.markdown(user_input)
    generate_assistant_response(user_input)

def generate_assistant_response(user_input):
    response = requests.post(BACKEND_QUERY_URL, json={"question": user_input})
    if response.status_code == 200:
        response_data = response.json()
        st.session_state.trace_id = response_data.get("trace_id", "No trace ID generated from backend.")
        assistant_response = response_data.get("response", "I'm sorry, I couldn't retrieve the information.")
        if isinstance(assistant_response, dict):
            assistant_response = assistant_response.get("output", "Sorry, I couldn't process that.")
    else:
        assistant_response = "Error: Unable to get response from the server."
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    st.rerun()

if __name__ == "__main__":
    st.set_page_config(page_title="Chat with Hälsa+GPT", page_icon="halsa_logo.png", layout="wide")
    top_bar = """
    <div style="display: flex; align-items: center; gap: 5px; margin-bottom: 20px;">
      <h1 style="margin: 0;">Chat with Hälsa+GPT</h1>
      <svg width="40" height="48" viewBox="0 0 40 48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M0 25.1429C0 22.6181 1.98985 20.5714 4.44444 20.5714H8.88889C11.3435 20.5714 13.3333 22.6181 13.3333 25.1429V29.7143C13.3333 32.239 11.3435 34.2857 8.88889 34.2857H4.44444C1.98985 34.2857 0 32.239 0 29.7143V25.1429Z" fill="white"/>
        <path d="M13.3333 11.4286C13.3333 8.90384 15.3232 6.85714 17.7778 6.85714H22.2222C24.6768 6.85714 26.6667 8.90384 26.6667 11.4286V16C26.6667 18.5247 24.6768 20.5714 22.2222 20.5714H17.7778C15.3232 20.5714 13.3333 18.5247 13.3333 16V11.4286Z" fill="white"/>
        <path d="M6.66667 2.28571C6.66667 1.02335 7.66159 0 8.88889 0H11.1111C12.3384 0 13.3333 1.02335 13.3333 2.28571V4.57143C13.3333 5.83379 12.3384 6.85714 11.1111 6.85714H8.88889C7.66159 6.85714 6.66667 5.83379 6.66667 4.57143V2.28571Z" fill="white"/>
        <path d="M2.22222 8C2.22222 7.36882 2.71968 6.85714 3.33333 6.85714H5.55556C6.16921 6.85714 6.66667 7.36882 6.66667 8V10.2857C6.66667 10.9169 6.16921 11.4286 5.55556 11.4286H3.33333C2.71968 11.4286 2.22222 10.9169 2.22222 10.2857V8Z" fill="white"/>
        <path d="M26.6667 25.1429C26.6667 22.6181 28.6565 20.5714 31.1111 20.5714H35.5556C38.0102 20.5714 40 22.6181 40 25.1429V29.7143C40 32.239 38.0102 34.2857 35.5556 34.2857H31.1111C28.6565 34.2857 26.6667 32.239 26.6667 29.7143V25.1429Z" fill="white"/>
        <path d="M13.3333 38.8571C13.3333 36.3324 15.3232 34.2857 17.7778 34.2857H22.2222C24.6768 34.2857 26.6667 36.3324 26.6667 38.8571V43.4286C26.6667 45.9533 24.6768 48 22.2222 48H17.7778C15.3232 48 13.3333 45.9533 13.3333 43.4286V38.8571Z" fill="white"/>
      </svg>
    </div>
    """
    st.markdown(top_bar, unsafe_allow_html=True)
    st.markdown("### Hi Anders,")
    st.markdown("Welcome to your personal AI assistant, Hälsa+GPT, which analyzes your health data in a secure \n\n GDPR- and HIPAA-compliant system.")
    main()