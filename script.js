const checkbox = document.getElementById('detei mode');
const imageContainer = document.getElementById('image-container');
const fullscreenImage = document.getElementById('fullscreen-image');
imageContainer.style.display = 'none';

checkbox.addEventListener('change', function () {
	if (this.checked) {
		// Показываем изображение
		imageContainer.style.display = 'flex';
		fullscreenImage.classList.remove('fade-out');

		setTimeout(() => {
			fullscreenImage.classList.add('fade-out');

			// После завершения анимации скрываем всё
			setTimeout(() => {
				imageContainer.style.display = 'none';
			}, 200);
		}, 100);
	}
});

let model;
const maxLen = 152; // Должно совпадать с max_sequence_len из обучения

// Загрузка модели
async function loadModel() {
	const model = await tf.loadLayersModel('tfjs_model/model.json');
	console.log("Модель загружена!");
	document.getElementById("next button").style.display = "block";
	document.getElementById("loading text").style.text = "Модель загружена"
}

// Генерация текста
async function generateText() {
	const inputText = document.getElementById('inputText').value;
	if (!model) {
		alert("Модель ещё загружается... Подождите немного.");
		return;
	}

	let output = inputText;
	const num_tokens = document.getElementById("num tokens");
	for (let i = 0; i < num_tokens; i++) {
		const tokenized = tokenizeText(output);
		const padded = padSequence(tokenized, maxLen);
		const prediction = model.predict(padded);
		const nextWord = getWordFromPrediction(prediction);
		output += " " + nextWord;
	}

	if (checkbox.checked) {
		output = output.replace("говном", "< УДАЛЕНО >")
		output = output.replace("Говно", "< УДАЛЕНО >")
		output = output.replace("говно", "< УДАЛЕНО >")
		output = output.replace("говнецо", "< УДАЛЕНО >")
		output = output.replace("говне", "< УДАЛЕНО >")
		output = output.replace("говна", "< УДАЛЕНО >")
		output = output.replace("говном", "< УДАЛЕНО >")
		output = output.replace("хитрожопая", "< УДАЛЕНО >")
		output = output.replace("жопа", "< УДАЛЕНО >")
		output = output.replace("хитрожопых", "< УДАЛЕНО >")
		output = output.replace("хитрожопые", "< УДАЛЕНО >")
		output = output.replace("поджопывать", "< УДАЛЕНО >")
		output = output.replace("жопы", "< УДАЛЕНО >")
		output = output.replace("жопе", "< УДАЛЕНО >")
		output = output.replace("жопы", "< УДАЛЕНО >")
		output = output.replace("посрать", "< УДАЛЕНО >")
		output = output.replace("высрать", "< УДАЛЕНО >")
		output = output.replace("срать", "< УДАЛЕНО >")
		output = output.replace("насрал", "< УДАЛЕНО >")
		output = output.replace("обосрал", "< УДАЛЕНО >")
		output = output.replace("сраки", "< УДАЛЕНО >")
		output = output.replace("дрочит", "< УДАЛЕНО >")
	}
	document.getElementById('output').innerText = output;
}

// Токенизация (упрощённая версия)
function tokenizeText(text) {
	return text.toLowerCase().split(' ');
}

// Загрузка модели при старте
loadModel();
document.getElementById('generateBtn').addEventListener('click', generateText);

// Скрытие окна ИИ
document.getElementById("AI menu").style.display = "none";
document.getElementById("next button").style.display = "none";

async function Show_AI_menu() {
	document.getElementById("AI menu").style.display = "block";
	document.getElementById("Start page").style.display = "none";
}