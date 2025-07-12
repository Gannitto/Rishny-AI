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
	const tokenizerFile = await fetch('./tfjs_model/tokenizer.json')
	console.log(tokenizerFile)
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
}
async function generateText() {
	const inputText = document.getElementById('inputText').value;
	

	document.getElementById('output').innerText = output;


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