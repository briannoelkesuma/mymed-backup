import os
import streamlit as st
import requests
import uuid

# Custom icons for user and assistant
user_icon = "https://cdn-icons-png.flaticon.com/512/6897/6897018.png"
assistant_icon = "https://mymedicaldata.se/wp-content/uploads/2022/12/Nylogga-web.png"

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

BACKEND_URL = "http://localhost:9009/query/"  # Replace with your backend URL

def main():
    initialise_chat()
    display_suggestions()
    display_chat_history()
    handle_user_input()


def initialise_chat():
    """Initialize the chat history and suggestions visibility."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "suggestions_visible" not in st.session_state:
        st.session_state.suggestions_visible = True


def display_suggestions():
    """Display suggestions for general and specific questions."""
    if st.session_state.suggestions_visible and not st.session_state.chat_history:
        st.markdown("### Need some ideas? Try these:")
        
        st.markdown("###### General Questions")
        display_suggestion_buttons(GENERAL_QUESTIONS, key_prefix="general")
        
        st.markdown("###### More Specific Questions")
        display_suggestion_buttons(SPECIFIC_QUESTIONS, key_prefix="specific")


def display_suggestion_buttons(questions, key_prefix):
    """Display buttons for suggestions."""
    for i, question in enumerate(questions):
        # Use a consistent key for each button based on its index and prefix
        button_key = f"{key_prefix}_question_{i}"
        if st.button(question, key=button_key):
            add_user_input(question)
            st.session_state.suggestions_visible = False
            st.rerun()


def display_chat_history():
    """Render chat history with appropriate roles and icons."""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user", avatar=user_icon):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar=assistant_icon):
                st.markdown(message["content"])


def handle_user_input():
    """Handle manual user input from the chat box."""
    user_input = st.chat_input("Ask me anything about your health based on your data…")
    if user_input:
        add_user_input(user_input)


def add_user_input(user_input):
    """Add user input to chat history and generate a response."""
    # Append user input to the chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=user_icon):
        st.markdown(user_input)

    # Generate and display assistant response
    generate_assistant_response(user_input)


def generate_assistant_response(user_input):
    """Generate assistant response by querying the backend."""
    with st.chat_message("assistant", avatar=assistant_icon):
        # Call the backend API
        response = requests.post(BACKEND_URL, json={"question": user_input})

        if response.status_code == 200:
            response_data = response.json()

            # Handle response output
            assistant_response = response_data.get("response", "I'm sorry, I couldn't retrieve the information.")
            if isinstance(assistant_response, dict):
                assistant_response = assistant_response.get("output", "Sorry, I couldn't process that.")

            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            st.markdown(assistant_response)
        else:
            st.markdown("Error: Unable to get response from the server.")


if __name__ == "__main__":
    st.set_page_config(page_title="Chat with Hälsa+GPT", page_icon="logo.png", layout="wide")
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

    # Render the SVG in Streamlit
    st.markdown(top_bar, unsafe_allow_html=True)
    st.markdown("### Hi Anders,")
    st.markdown("Welcome to your personal AI assistant, Hälsa+GPT, which analyzes your health data in a secure \n\n GDPR- and HIPAA-compliant system.")
    main()