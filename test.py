import pickle
import numpy as np
import onnxruntime as ort
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Загрузка токенизатора

def load_rishny_ai():
	session = ort.InferenceSession("model.onnx")
	with open("tokenizer.pickle", "rb") as f:
		tokenizer = pickle.load(f)  # Загружаем токенизатор
	print(type(tokenizer))  # Определяем класс токенизатора
	print(dir(tokenizer))   # Смотрим доступные методы
	return session, tokenizer

session, tokenizer = load_rishny_ai()
print("Модель загружена!")

def prepare_input(text: str):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=152, padding='post')
    return {"input": padded.astype(np.float32)}

def generate_text(text):
	# 1. Токенизация входного текста (адаптируйте под вашу модель!)
	input_ids = prepare_input(text)
	
	# 2. Генерация текста
	generated_text = generate(input_ids)
	
	return {"generated_text": generated_text}

# --- Вспомогательные функции ---


def generate(input_ids: np.ndarray, max_length: int=10) -> str:
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
	print(generate_text("Ришный"))