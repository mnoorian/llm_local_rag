import streamlit as st
import requests
import os

st.title("ASK LLM (Conversational & Pure)")

# Service selection with clear label
endpoint_options = {
    "Conversational RAG (Memory)": {
        "url": os.getenv("API_URL_RAG", "http://backend:8000/rag_query"),
        "description": "Conversational memory, context-aware answers"
    },
    "Pure LLM (No Memory)": {
        "url": os.getenv("API_URL_LLM", "http://backend:8000/generate"),
        "description": "Single-turn, stateless LLM response"
    }
}

with st.form(key="llm_form"):
    service_type = st.selectbox(
        "Select Service Type (Endpoint)",
        list(endpoint_options.keys()),
        format_func=lambda x: f"{x}"
    )
    st.caption(f"**Endpoint:** `{endpoint_options[service_type]['url']}` — {endpoint_options[service_type]['description']}")
    prompt = st.text_area("Enter your question:")
    submit = st.form_submit_button("Ask")

# Session state for session_id and conversation history
if "last_service_type" not in st.session_state:
    st.session_state.last_service_type = service_type
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Reset session state if endpoint changes
if service_type != st.session_state.last_service_type:
    st.session_state.session_id = None
    st.session_state.conversation_history = []
    st.session_state.last_service_type = service_type


def ask_llm(prompt: str, session_id: str = None):
    try:
        api_url = endpoint_options[service_type]["url"]
        if service_type == "Conversational RAG (Memory)":
            payload = {"query": prompt}
            if session_id:
                payload["session_id"] = session_id
        else:
            payload = {"prompt": prompt}
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            data = response.json()
            if service_type == "Conversational RAG (Memory)":
                return data.get("response", "No answer returned."), data.get("session_id"), data.get("conversation_history", [])
            else:
                return data.get("response", "No answer returned."), None, []
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
    return None, None, None

if submit:
    if prompt.strip():
        with st.spinner("Generating answer..."):
            answer, session_id, conversation_history = ask_llm(prompt, st.session_state.session_id)
            if answer:
                if service_type == "Conversational RAG (Memory)":
                    st.session_state.session_id = session_id
                    st.session_state.conversation_history = conversation_history
                st.success(answer)
    else:
        st.warning("Please enter a question.")

# Display conversation history only for Conversational RAG
if service_type == "Conversational RAG (Memory)" and st.session_state.conversation_history:
    st.markdown("---")
    st.subheader("Conversation History")
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")

# Option to reset conversation (only for Conversational RAG)
if service_type == "Conversational RAG (Memory)":
    if st.button("Reset Conversation"):
        st.session_state.session_id = None
        st.session_state.conversation_history = []
        st.success("Conversation reset.") 