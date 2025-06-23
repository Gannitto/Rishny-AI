import pickle
from fastapi import FastAPI
import numpy as np
import onnxruntime as ort
from pydantic import BaseModel
import logging
from fastapi.middleware.cors import CORSMiddleware

# Загрузка токенизатора
with open('tokenizer.pickle', 'rb') as handle:
	tokenizer = pickle.load(handle)

app = FastAPI()
logger = logging.getLogger(__name__)
logger.info("Запуск..")
session = ort.InferenceSession("model.onnx")

pp = FastAPI()

# Настройка CORS
app.add_middleware(
	CORSMiddleware,
	allow_origins=["https://gannitto.github.io"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"]
)
try:
	
	# Класс для входных данных API
	class TextRequest(BaseModel):
		text: str
		max_length: int = 50  # Максимальная длина генерируемого текста

	def tokenize_text(text: str) -> np.ndarray:
		"""Преобразует текст в вектор (пример для моделей типа LSTM/Transformer)."""
		return np.array([[1, 2, 3]], dtype=np.int64)  # Заглушка

	def prepare_input(text: str):
		# Токенизация текста
		tokens = tokenizer.encode(text)
	
		# Приведение к нужной длине (152)
		tokens = tokens[:152] + [0] * (152 - len(tokens))  # Паддинг нулями
	
		# Создание тензора с явным указанием float32
		input_tensor = np.array([tokens], dtype=np.float32)  # Форма: [1, 152]
	
		print(f"Подготовленный тензор - форма: {input_tensor.shape}, тип: {input_tensor.dtype}")
		return {"input": input_tensor}

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
	
	@app.post("/generate")
	async def generate_text(request: TextRequest):
		try:
			inputs = prepare_input(request.text)
		
			# generated_text = generate(
			# 	session,
			# 	input_ids,
			# 	max_length=request.max_length
			# )
			outputs = session.run(None, inputs)
		

			return {"generated_text": str(outputs[0].tolist())}
	
		except Exception as e:
			return {"error": type(e).__name__ + "\n" + str(e)}
		
	if __name__ == "__main__":
		import uvicorn

		uvicorn.run(app, host="0.0.0.0", port=8000)

except Exception as e:
	# Обработка ошибки
	error_name = type(e).__name__
	error_message = str(e)
	logger.info("Упс, ошибка")
	logger.info(f"Ошибка: {error_name},  Сообщение: {error_message}")
