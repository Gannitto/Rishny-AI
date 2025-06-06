let model;
const maxLen = 20; // ������ ��������� � max_sequence_len �� ��������

// �������� ������
async function loadModel() {
	const model = await tf.loadLayersModel('tfjs_model/model.json');
	const inputShape = [null, 152];
	model.layers[0].batchInputShape = inputShape;
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