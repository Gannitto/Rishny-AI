let model;
const maxLen = 20; // Должно совпадать с max_sequence_len из обучения

// Загрузка модели
async function loadModel() {
	const model = await tf.loadLayersModel('tfjs_model/model.json');
	const inputShape = [null, 152];
	model.layers[0].batchInputShape = inputShape;
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