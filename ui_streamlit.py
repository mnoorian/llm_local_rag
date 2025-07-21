import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/generate")

st.title("ASK LLM")

prompt = st.text_area("Enter your question:")

def ask_llm(prompt: str):
    try:
        response = requests.post(
            API_URL,
            json={"prompt": prompt}
        )
        if response.status_code == 200:
            return response.json().get("response", "No answer returned.")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
    return None

if st.button("Ask"):
    if prompt.strip():
        with st.spinner("Generating answer..."):
            answer = ask_llm(prompt)
            if answer:
                st.success(answer)
    else:
        st.warning("Please enter a question.") 