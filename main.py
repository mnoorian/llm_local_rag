from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import os

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    response: str

def create_app():
    MODEL_PATH = os.getenv("MISTRAL_MODEL_PATH", "./mistral-7b-instruct-v0.2.Q2_K.gguf")
    try:
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=os.cpu_count(), n_gpu_layers=0)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest):
        try:
            output = llm(
                req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                stop=None,
                echo=False
            )
            return GenerateResponse(response=output["choices"][0]["text"].strip())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app

app = create_app() 