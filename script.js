const checkbox = document.getElementById('detei mode');
const imageContainer = document.getElementById('image-container');
const fullscreenImage = document.getElementById('fullscreen-image');
let sequenceLength = 5;
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
	const tokenizerFile = await fetch('./tfjs_model/tokenizer.json')
	const tokenizerText = await tokenizerFile.text();
	const tokenizerData = JSON.parse(tokenizerText);

	loadedTokenizer = {
		wordIndex: tokenizerData.wordIndex,
		indexWord: tokenizerData.indexWord,
		numWords: tokenizerData.numWords,
		textsToSequences: function (texts) {
			return texts.map(text =>
				text.toLowerCase().split(' ')
					.map(word => this.wordIndex[word])
					.filter(idx => idx !== undefined)
			);
		}
	};
	console.log("Model loaded!");
	console.log('Модель загружена:', model);
	console.log('Токенизатор загружен:', loadedTokenizer);
}
function generateText(Model, tokenizer, seedText, length = 20) {
	let result = seedText;
	let currentSeq = seedText.toLowerCase().split(/\s+/).filter(word => word.length > 0);

	if (currentSeq.length > sequenceLength) {
		currentSeq = currentSeq.slice(-sequenceLength);
	}

	for (let i = 0; i < length; i++) {
		const inputSeq = currentSeq.map(word => tokenizer.wordIndex[word] || 0);

		while (inputSeq.length < sequenceLength) {
			inputSeq.unshift(0);
		}

		const inputTensor = tf.tensor2d([inputSeq], [1, sequenceLength], 'float32');
		const output = Model.predict(inputTensor);
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
		result = result.replace("говном", "< УДАЛЕНО >")
		result = result.replace("Говно", "< УДАЛЕНО >")
		result = result.replace("говно", "< УДАЛЕНО >")
		result = result.replace("говнецо", "< УДАЛЕНО >")
		result = result.replace("говне", "< УДАЛЕНО >")
		result = result.replace("говна", "< УДАЛЕНО >")
		result = result.replace("говном", "< УДАЛЕНО >")
		result = result.replace("хитрожопая", "< УДАЛЕНО >")
		result = result.replace("жопа", "< УДАЛЕНО >")
		result = result.replace("хитрожопых", "< УДАЛЕНО >")
		result = result.replace("хитрожопые", "< УДАЛЕНО >")
		result = result.replace("поджопывать", "< УДАЛЕНО >")
		result = result.replace("жопы", "< УДАЛЕНО >")
		result = result.replace("жопе", "< УДАЛЕНО >")
		result = result.replace("жопы", "< УДАЛЕНО >")
		result = result.replace("посрать", "< УДАЛЕНО >")
		result = result.replace("высрать", "< УДАЛЕНО >")
		result = result.replace("срать", "< УДАЛЕНО >")
		result = result.replace("насрал", "< УДАЛЕНО >")
		result = result.replace("обосрал", "< УДАЛЕНО >")
		result = result.replace("сраки", "< УДАЛЕНО >")
		result = result.replace("дрочит", "< УДАЛЕНО >")
	}
	document.getElementById('output').innerText = result;

	return output;
}

// Загрузка модели при старте
loadModel();
document.getElementById('generateBtn').addEventListener('click', () => {
	try {
		console.log(model, loadedTokenizer, document.getElementById('inputText').value)
		const text = generateText(model, loadedTokenizer, document.getElementById('inputText').value, 20);
		//newOutput.innerHTML = text;
	} catch (error) {
		console.error('Ошибка генерации:', error);
		document.getElementById('output').innerHTML = 'Ошибка генерации: ' + error.message;
	}
});