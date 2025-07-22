# Mistral 7B Instruct v0.2 Local LLM API & UI

## 🚀 Quick Start with Docker Compose

1. **Download the model file**
   - Download `mistral-7b-instruct-v0.2.Q2_K.gguf` from [TheBloke's HuggingFace page](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF).
   - Place it in your project directory: `llm_local_rag/`.
2. **Build the Docker images**
   ```bash
   docker-compose build
   ```
3. **Start the services**
   ```bash
   docker-compose up
   ```
4. **Access the app**
   - Streamlit UI: [http://localhost:8501](http://localhost:8501)
   - FastAPI backend: [http://localhost:8000/generate](http://localhost:8000/generate)

---

This project runs the Mistral 7B Instruct v0.2 model locally (CPU-only) using llama.cpp (via llama-cpp-python), exposes a FastAPI server with a `/generate` endpoint, and provides a simple Streamlit UI for interaction.

## Requirements
- Python 3.8+ (for local runs)
- Docker & Docker Compose (for containerized runs)
- CPU with sufficient RAM (at least 8GB recommended)

## Setup

### 1. Download the Quantized GGUF Model (4-bit)
- Get the file `mistral-7b-instruct-v0.2.Q2_K.gguf` from [TheBloke's HuggingFace page](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) or another trusted source.
- Place it in the project directory (`llm_local_rag/`).

## Running with Docker Compose (Recommended)

1. **Build the images** (model file not required for this step):
   ```bash
   docker-compose build
   ```
2. **Ensure the model file is present** in your project directory:
   - `llm_local_rag/mistral-7b-instruct-v0.2.Q2_K.gguf`
3. **Start the services:**
   ```bash
   docker-compose up
   ```
   - The backend will mount the model file at runtime.
4. **Access the application:**
   - Streamlit UI: [http://localhost:8501](http://localhost:8501)
   - FastAPI backend: [http://localhost:8000/generate](http://localhost:8000/generate)

---

## Running Locally (Without Docker)

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the FastAPI backend:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
3. **Start the Streamlit UI (in a new terminal):**
   ```bash
   streamlit run ui_streamlit.py
   ```

---

## Usage
- Open the Streamlit UI in your browser: [http://localhost:8501](http://localhost:8501)
- Type your question and click "Ask". The UI will call the FastAPI backend and display the answer.

### API Usage (Direct)
Send a POST request to `/generate` with a JSON body:
```json
{
  "prompt": "What is the capital of France?",
  "max_tokens": 128,
  "temperature": 0.7
}
```
Response:
```json
{
  "response": "The capital of France is Paris."
}
```
- `max_tokens` and `temperature` are optional.

## Notes
- For best performance, use a quantized 4-bit GGUF model.
- Adjust `MISTRAL_MODEL_PATH` if your model file is elsewhere.
- The model runs entirely on CPU.

---

**Enjoy running your own local LLM with a simple web UI!** 

Your error means the FastAPI backend (LLM API) is not running or crashed, so the Streamlit UI cannot connect to it.

From your previous logs, the FastAPI server crashed with a segmentation fault (`zsh: segmentation fault  uvicorn main:app --host 0.0.0.0 --port 8000`). This is likely due to one of these reasons:
- The model file is too large for your available RAM.
- There is a bug or incompatibility in the llama-cpp-python or GGUF file.
- The quantization or model file is corrupted or not supported.

**How to troubleshoot:**

1. **Check RAM usage:**  
   Make sure you have enough free RAM (8GB+ recommended for 7B models, more for larger quantizations).

2. **Try a smaller quantized model:**  
   If you have a Q4 or Q5 model, try Q4_K_M or Q4_0 quantization, which uses less RAM.

3. **Update llama-cpp-python:**  
   Make sure you have the latest version:
   ```bash
   pip install --upgrade llama-cpp-python
   ```

4. **Test model loading in Python shell:**  
   Try this in a Python shell to see if the model loads without FastAPI:
   ```python
   from llama_cpp import Llama
   llm = Llama(model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=2048, n_threads=4, n_gpu_layers=0)
   ```
   If this crashes, the issue is with the model or llama-cpp-python.

5. **Check model file:**  
   Make sure your GGUF file is not corrupted and is the correct quantization for your hardware.

---

Would you like to try loading the model in a Python shell to see if it works, or do you want to try updating llama-cpp-python first? Let me know how you want to proceed! 

---

### 1. **Test Model Loading in Python Shell**

This will help determine if the problem is with the model file or the FastAPI integration.

Open a Python shell in your project directory and run:
```python
from llama_cpp import Llama
llm = Llama(model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=2048, n_threads=4, n_gpu_layers=0)
```
- If this **crashes** or gives a segmentation fault, the issue is with the model file or llama-cpp-python.
- If it **loads successfully**, the problem may be with how FastAPI is using it.

---

### 2. **If It Crashes:**

- **Try a smaller quantized model** (e.g., Q4_K_M or Q4_0).
- **Update llama-cpp-python** to the latest version:
  ```bash
  pip install --upgrade llama-cpp-python
  ```
- **Check your RAM usage**: Make sure you have enough free memory.

---

### 3. **If It Loads Successfully:**

- Try running your FastAPI server again:
  ```bash
  uvicorn main 
