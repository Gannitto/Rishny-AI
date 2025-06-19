from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np

app = FastAPI()

# Разрешить CORS (для запросов из браузера)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["POST"],
)

model = tf.keras.models.load_model("your_model.h5")

@app.post("/predict")
async def predict(data: dict):
	try:
		input_tensor = np.array(data["data"], dtype=np.float32).reshape(1, 224, 224, 3)
		predictions = model.predict(input_tensor)
		return {"predictions": predictions.tolist()}
	except Exception as e:
		return {"error": str(e)}