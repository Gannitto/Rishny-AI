from fastapi import FastAPI
import numpy as np
import onnxruntime as ort
from pydantic import BaseModel
import logging
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
logger = logging.getLogger(__name__)
logger.info("Запуск..")

try:
	session = ort.InferenceSession("model.onnx")

	app = FastAPI()

	# Настройка CORS
	app.add_middleware(
		CORSMiddleware,
		allow_origins=["https://gannitto.github.io"],
		allow_methods=["POST", "OPTIONS"],
		allow_headers=["Content-Type"]
	)

	# Класс для входных данных API
	class TextRequest(BaseModel):
		text: str
		max_length: int = 50  # Максимальная длина генерируемого текста

	@app.post("/generate")
	async def generate_text(request: TextRequest):
		try:
			input_ids = tokenize_text(request.text)
			input_ids = input_ids.astype(np.float32)
		
			# 2. Генерация текста
			generated_text = generate(
				session,
				input_ids,
				max_length=request.max_length
			)
		
			return {"generated_text": generated_text}
	
		except Exception as e:
			return {"error": str(e)}

	# --- Вспомогательные функции ---

	def tokenize_text(text: str) -> np.ndarray:
		"""Преобразует текст в вектор (пример для моделей типа LSTM/Transformer)."""
		# Пример: если модель ожидает вход shape=(1, seq_len)
		# Замените на ваш токенизатор!
		return np.array([[1, 2, 3]], dtype=np.int64)  # Заглушка

	def generate(session: ort.InferenceSession, input_ids: np.ndarray, max_length: int) -> str:
		"""Генерирует текст с помощью ONNX-модели."""
		generated = []
		for _ in range(max_length):
			# Предсказание следующего токена
			outputs = session.run(
				None,
				{"input": input_ids}  # Имя входного слоя (см. через Netron)
			)
			next_token = np.argmax(outputs[0][0, -1])  # Пример для классических моделей
		
			# Обновляем вход для следующего шага
			input_ids = np.concatenate(
				[input_ids, np.array([[next_token]], dtype=np.int64)],
				axis=1
			)
			generated.append(next_token)
		
			# Остановка по спецтокену (например, конец текста)
			if next_token == 2:  # Пример для токена </s>
				break
	
		# Детокенизация (замените на вашу логику!)
		return " ".join(map(str, generated))  # Заглушка

	if __name__ == "__main__":
		import uvicorn

		uvicorn.run(app, host="0.0.0.0", port=8000)

except Exception as e:
	# Обработка ошибки
	error_name = type(e).__name__
	error_message = str(e)
	logger.info("Упс, ошибка")
	logger.info(f"Ошибка: {error_name},  Сообщение: {error_message}")
