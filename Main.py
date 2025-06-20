from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from ai_model import load_rishny_ai, generate_text

app = FastAPI()
load_rishny_ai()

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["POST"],
)

@app.post("/predict")
async def predict(data: dict):
	try:
		input_text = data["text"]
		next_words = data["next_words"]
		result = generate_text(input_text, next_words)
		return {"result": result}
	except Exception as e:
		return {"error": str(e)}