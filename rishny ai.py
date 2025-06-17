import tensorflow
from tensorflow.keras.models import load_model
import pickle
from js import document
from pyodide import create_proxy

model = load_model("rishny_model.h5")  # Загружаем модель
with open("tokenizer.pickle", "rb") as f:
	tokenizer = pickle.load(f)  # Загружаем токенизатор

document.getElementById("next button").style.display = "block"

def generate_text(event):

	seed_text = document.getElementById("inputText").value
	next_words = document.getElementById("nextWords").value
	for _ in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")
		predicted = np.argmax(model.predict(token_list), axis=-1)
		output_word = ""
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word

	# Делаем первую букву каждого предложения заглавной
	pattern = r'(?:[.!?…]+)\s+([a-zа-я])'
	# Заменяем строчную букву после знаков препинания на заглавную
	def replacer(match):
		return match.group(0).upper()  # заменяем следующую букву на заглавную

	# Используем re.sub с функцией замены
	corrected_text = sub(
		pattern,
		lambda m: m.group(0)[:-1] + m.group(0)[-1].upper(),  # последний символ (буква) становится заглавным
		seed_text
	)

	# Также делаем первую букву всего текста заглавной
	if corrected_text:
		corrected_text = corrected_text[0].upper() + corrected_text[1:]

	# Добавляем точку в конце, если нужно
	if not (corrected_text[-1] in [".", "!", "?"]):
		corrected_text += "."

	document.getElementById("output").innerText = corrected_text

button = document.getElementById("generateBtn")
button.addEventListener("click", create_proxy(generate_text))