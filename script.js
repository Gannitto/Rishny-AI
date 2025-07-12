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
	console.log("Model loaded!");
}

// Генерация текста
async function generateText() {

	const inputText = document.getElementById('inputText').value;
	//if (!model) {
	//	alert("Модель ещё загружается... Подождите немного.");
	//	return;
	//}

	let output = inputText;
	const num_tokens = document.getElementById("nextWords").value;
	

	const tokenizer = {
		wordIndex: {},
		indexWord: {},
		wordCounts: {},
		numWords: 0,
		textsToSequences: function (texts) {
			return texts.map(text =>
				text.toLowerCase().split(' ')
					.map(word => this.wordIndex[word])
					.filter(idx => idx !== undefined)
			);
		}
	}

	let sequenceLength = 5;
	let result = inputText;
	let currentSeq = inputText.toLowerCase().split(/\s+/).filter(word => word.length > 0);

	if (currentSeq.length > sequenceLength) {
		currentSeq = currentSeq.slice(-sequenceLength);
	}

	for (let i = 0; i < num_tokens; i++) {
		//const tokenized = tokenizeText(output);
		//const padded = padSequence(tokenized, maxLen);
		//const prediction = model.predict(padded);
		//const nextWord = getWordFromPrediction(prediction);
		//output += " " + nextWord;
		const inputSeq = currentSeq.map(word => tokenizer.wordIndex[word] || 0);

		while (inputSeq.length < sequenceLength) {
			inputSeq.unshift(0);
		}

		const inputTensor = tf.tensor2d([inputSeq], [1, sequenceLength], 'float32');
		const output = model.predict(inputTensor);
		const nextIndex = tf.argMax(output, -1).dataSync()[0];
		inputTensor.dispose();
		output.dispose();

		const nextWord = tokenizer.indexWord[nextIndex];
		if (!nextWord) break;

		result += ' ' + nextWord;
		currentSeq.push(nextWord);
		if (currentSeq.length > sequenceLength) {
			currentSeq.shift();
		}
	}

	if (checkbox.checked) {
		output = output.replace("говном", "< УДАЛЕНО >")
		output = output.replace("Говно", "< УДАЛЕНО >")
		output = output.replace("говно", "< УДАЛЕНО >")
	}
	document.getElementById('output').innerText = output;
}

// Токенизация
function tokenizeText(text) {
	return text.toLowerCase().split(' ');
}

// Загрузка модели при старте
loadModel();
document.getElementById('generateBtn').addEventListener('click', generateText);