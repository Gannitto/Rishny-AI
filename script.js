const checkbox = document.getElementById('detei mode');
const imageContainer = document.getElementById('image-container');
const fullscreenImage = document.getElementById('fullscreen-image');
imageContainer.style.display = 'none';

checkbox.addEventListener('change', function () {
	if (this.checked) {
		// ���������� �����������
		imageContainer.style.display = 'flex';
		fullscreenImage.classList.remove('fade-out');

		setTimeout(() => {
			fullscreenImage.classList.add('fade-out');

			// ����� ���������� �������� �������� ��
			setTimeout(() => {
				imageContainer.style.display = 'none';
			}, 200);
		}, 100);
	}
});

let model;
const maxLen = 152; // ������ ��������� � max_sequence_len �� ��������

// �������� ������
async function loadModel() {
	const model = await tf.loadLayersModel('tfjs_model/model.json');
	console.log("������ ���������!");
	document.getElementById("next button").style.display = "block";
	document.getElementById("loading text").style.text = "������ ���������"
}

// ��������� ������
async function generateText() {
	const inputText = document.getElementById('inputText').value;
	if (!model) {
		alert("������ ��� �����������... ��������� �������.");
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
		output = output.replace("������", "< ������� >")
		output = output.replace("�����", "< ������� >")
		output = output.replace("�����", "< ������� >")
		output = output.replace("�������", "< ������� >")
		output = output.replace("�����", "< ������� >")
		output = output.replace("�����", "< ������� >")
		output = output.replace("������", "< ������� >")
		output = output.replace("����������", "< ������� >")
		output = output.replace("����", "< ������� >")
		output = output.replace("����������", "< ������� >")
		output = output.replace("����������", "< ������� >")
		output = output.replace("�����������", "< ������� >")
		output = output.replace("����", "< ������� >")
		output = output.replace("����", "< ������� >")
		output = output.replace("����", "< ������� >")
		output = output.replace("�������", "< ������� >")
		output = output.replace("�������", "< ������� >")
		output = output.replace("�����", "< ������� >")
		output = output.replace("������", "< ������� >")
		output = output.replace("�������", "< ������� >")
		output = output.replace("�����", "< ������� >")
		output = output.replace("������", "< ������� >")
	}
	document.getElementById('output').innerText = output;
}

// ����������� (���������� ������)
function tokenizeText(text) {
	return text.toLowerCase().split(' ');
}

// �������� ������ ��� ������
loadModel();
document.getElementById('generateBtn').addEventListener('click', generateText);

// ������� ���� ��
document.getElementById("AI menu").style.display = "none";
document.getElementById("next button").style.display = "none";

async function Show_AI_menu() {
	document.getElementById("AI menu").style.display = "block";
	document.getElementById("Start page").style.display = "none";
}