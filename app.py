from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Разрешить запросы с фронтенда

# Загружаем модель
model = tf.keras.models.load_model("rishny_model.keras")

@app.route("/predict", methods=["POST"])
def predict():
	try:
		# Получаем данные от клиента (например, изображение в base64 или массив)
		data = request.json["data"]
		
		# Преобразуем данные в тензор (пример для модели с входом [1, 224, 224, 3])
		import numpy as np
		input_tensor = np.array(data, dtype=np.float32).reshape(1, 224, 224, 3)
		
		# Предсказание
		predictions = model.predict(input_tensor)
		
		# Возвращаем результат
		return jsonify({"predictions": predictions.tolist()})
	
	except Exception as e:
		return jsonify({"error": str(e)})

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000)