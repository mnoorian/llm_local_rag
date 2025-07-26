import streamlit as st
import requests
import os
import json
from datetime import datetime
import io

st.set_page_config(
    page_title="Agentic RAG for Financial Risk Analysis",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG for Financial Risk Analysis")
st.markdown("**Intelligent assistant for financial risk analysts with document retrieval, web search, and reasoning capabilities**")

# Service selection
endpoint_options = {
    "Agentic RAG (Advanced)": {
        "url": os.getenv("API_URL_AGENTIC", "http://backend:8000/agentic_query"),
        "description": "Multi-tool agentic system with reasoning, citations, and web search"
    },
    "Conversational RAG (Memory)": {
        "url": os.getenv("API_URL_RAG", "http://backend:8000/rag_query"),
        "description": "Conversational memory, context-aware answers"
    },
    "Pure LLM (No Memory)": {
        "url": os.getenv("API_URL_LLM", "http://backend:8000/generate"),
        "description": "Single-turn, stateless LLM response"
    }
}

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    service_type = st.selectbox(
        "Select Service Type",
        list(endpoint_options.keys()),
        format_func=lambda x: f"{x}"
    )
    
    st.caption(f"**Endpoint:** `{endpoint_options[service_type]['url']}`")
    st.caption(f"**Description:** {endpoint_options[service_type]['description']}")
    
    if service_type == "Agentic RAG (Advanced)":
        st.subheader("🔧 Agentic Settings")
        use_web_search = st.checkbox("Use Web Search", value=True, help="Enable web search for current information")
        use_document_search = st.checkbox("Use Document Search", value=True, help="Search regulatory documents")
        reasoning_steps = st.slider("Reasoning Steps", 2, 5, 3, help="Number of reasoning steps to generate")
        max_tokens = st.slider("Max Tokens", 256, 1024, 512, help="Maximum response length")
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, help="Response creativity (lower = more focused)")
    
    elif service_type == "Conversational RAG (Memory)":
        st.subheader("🔧 RAG Settings")
        max_tokens = st.slider("Max Tokens", 256, 1024, 256, help="Maximum response length")
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, help="Response creativity")
    
    else:  # Pure LLM
        st.subheader("🔧 LLM Settings")
        max_tokens = st.slider("Max Tokens", 256, 1024, 256, help="Maximum response length")
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, help="Response creativity")

    # Document upload section
    st.markdown("---")
    st.subheader("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload financial documents (PDF, TXT)",
        type=['pdf', 'txt'],
        help="Upload regulatory documents, reports, or other financial materials"
    )
    
    if uploaded_file is not None:
        if st.button("📤 Upload & Index"):
            with st.spinner("Uploading and indexing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post("http://backend:8000/upload_document", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.info(f"📊 Indexed {result['chunks']} document chunks")
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Upload error: {str(e)}")

# Session state management
if "last_service_type" not in st.session_state:
    st.session_state.last_service_type = service_type
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "agentic_responses" not in st.session_state:
    st.session_state.agentic_responses = []

# Reset session state if endpoint changes
if service_type != st.session_state.last_service_type:
    st.session_state.session_id = None
    st.session_state.conversation_history = []
    st.session_state.agentic_responses = []
    st.session_state.last_service_type = service_type

# Main query interface
with st.form(key="query_form"):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_area(
            "Enter your financial risk analysis query:",
            placeholder="e.g., What are the capital requirements under Basel III for credit risk?",
            height=100
        )
    
    with col2:
        st.write("")
        st.write("")
        submit = st.form_submit_button("🚀 Analyze", use_container_width=True)

def ask_agentic_rag(prompt: str, session_id: str = None):
    """Send query to agentic RAG endpoint"""
    try:
        api_url = endpoint_options[service_type]["url"]
        payload = {
            "query": prompt,
            "session_id": session_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "use_web_search": use_web_search,
            "use_document_search": use_document_search,
            "reasoning_steps": reasoning_steps
        }
        
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
    return None

def ask_rag(prompt: str, session_id: str = None):
    """Send query to conversational RAG endpoint"""
    try:
        api_url = endpoint_options[service_type]["url"]
        payload = {
            "query": prompt,
            "session_id": session_id,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
    return None

def ask_llm(prompt: str):
    """Send query to basic LLM endpoint"""
    try:
        api_url = endpoint_options[service_type]["url"]
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
    return None

# Process query submission
if submit:
    if query.strip():
        with st.spinner("🤖 Analyzing your query..."):
            if service_type == "Agentic RAG (Advanced)":
                result = ask_agentic_rag(query, st.session_state.session_id)
                if result:
                    st.session_state.session_id = result.get("session_id")
                    st.session_state.agentic_responses.append(result)
                    
                    # Display response
                    st.success("✅ Analysis Complete!")
                    
                    # Main response
                    st.subheader("📊 Analysis Result")
                    st.markdown(result.get("response", "No response generated."))
                    
                    # Confidence score
                    confidence = result.get("confidence_score", 0)
                    st.metric("Confidence Score", f"{confidence:.2%}")
                    
                    # Tools used
                    tools_used = result.get("tools_used", [])
                    if tools_used:
                        st.write("🔧 **Tools Used:**", ", ".join(tools_used))
                    
                    # Reasoning chain
                    reasoning_chain = result.get("reasoning_chain", [])
                    if reasoning_chain:
                        with st.expander("🧠 Reasoning Chain", expanded=False):
                            for i, step in enumerate(reasoning_chain, 1):
                                st.write(f"**Step {i}:** {step}")
                    
                    # Citations
                    citations = result.get("citations", [])
                    if citations:
                        with st.expander("📚 Sources & Citations", expanded=False):
                            for i, citation in enumerate(citations, 1):
                                with st.container():
                                    col1, col2 = st.columns([1, 4])
                                    with col1:
                                        st.write(f"**{i}.**")
                                    with col2:
                                        st.write(f"**{citation['source']}** ({citation['source_type']})")
                                        st.write(f"Relevance: {citation['relevance_score']:.2%}")
                                        st.write(f"*{citation['content'][:200]}...*")
                    
            elif service_type == "Conversational RAG (Memory)":
                result = ask_rag(query, st.session_state.session_id)
                if result:
                    st.session_state.session_id = result.get("session_id")
                    st.session_state.conversation_history = result.get("conversation_history", [])
                    st.success(result.get("response", "No response generated."))
            
            else:  # Pure LLM
                result = ask_llm(query)
                if result:
                    st.success(result.get("response", "No response generated."))
    else:
        st.warning("Please enter a query.")

# Display conversation history
if service_type == "Agentic RAG (Advanced)" and st.session_state.agentic_responses:
    st.markdown("---")
    st.subheader("📊 Analysis History")
    
    for i, response in enumerate(reversed(st.session_state.agentic_responses[-5:]), 1):
        with st.expander(f"Analysis {len(st.session_state.agentic_responses) - i + 1}: {response.get('conversation_history', [{}])[-2]['content'][:50]}...", expanded=False):
            st.markdown(f"**Query:** {response.get('conversation_history', [{}])[-2]['content']}")
            st.markdown(f"**Response:** {response.get('response', 'No response')}")
            st.write(f"**Confidence:** {response.get('confidence_score', 0):.2%}")
            st.write(f"**Tools:** {', '.join(response.get('tools_used', []))}")

elif service_type == "Conversational RAG (Memory)" and st.session_state.conversation_history:
    st.markdown("---")
    st.subheader("💬 Conversation History")
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")

# Reset conversation option
if service_type in ["Agentic RAG (Advanced)", "Conversational RAG (Memory)"]:
    if st.button("🔄 Reset Conversation"):
        st.session_state.session_id = None
        st.session_state.conversation_history = []
        st.session_state.agentic_responses = []
        st.success("Conversation reset successfully!")

# Footer with example queries
with st.expander("💡 Example Queries for Financial Risk Analysis"):
    st.markdown("""
    **Regulatory Compliance:**
    - What are the capital requirements under Basel III for credit risk?
    - How does Dodd-Frank affect proprietary trading?
    - What are the liquidity requirements under Solvency II?
    
    **Risk Assessment:**
    - How should I assess counterparty credit risk in derivatives trading?
    - What are the key risk indicators for operational risk?
    - How do I calculate Value at Risk (VaR) for a portfolio?
    
    **Market Analysis:**
    - What are the current trends in interest rate risk management?
    - How do I evaluate market risk in emerging markets?
    - What are the best practices for stress testing?
    """)

# System status
try:
    health_response = requests.get("http://backend:8000/health")
    if health_response.status_code == 200:
        health_data = health_response.json()
        if health_data.get("weaviate_connected"):
            st.sidebar.success("✅ Backend + Weaviate Connected")
        else:
            st.sidebar.warning("⚠️ Backend Connected (Weaviate: Fallback Mode)")
    else:
        st.sidebar.error("❌ Backend Error")
except:
    st.sidebar.error("❌ Backend Unavailable") 