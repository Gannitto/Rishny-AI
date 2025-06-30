const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node'); // Для использования CPU/GPU

// Пример данных: текст и соответствующие метки (0 - негативный, 1 - позитивный)
const texts = [
    "Я люблю этот продукт, он отличный!",
    "Ужасное качество, очень разочарован.",
    "Неплохо, но могло быть лучше.",
    "Прекрасно работает, рекомендую всем!",
    "Совершенно не оправдал моих ожиданий."
];
const labels = [1, 0, 0, 1, 0]; // Соответствующие метки

// Функция для токенизации текста
function tokenizeText(texts, maxWords = 1000) {
    const tokenizer = new tf.layers.TextVectorization({
        maxTokens: maxWords,
        outputMode: 'int',
        outputSequenceLength: 10 // Фиксированная длина последовательности
    });
    tokenizer.adapt(texts);
    return tokenizer;
}

// Создание модели
function createModel(vocabSize) {
    const model = tf.sequential();
    
    model.add(tf.layers.embedding({
        inputDim: vocabSize,
        outputDim: 16,
        inputLength: 10
    }));
    
    model.add(tf.layers.globalAveragePooling1d());
    
    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu'
    }));
    
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));
    
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    return model;
}

// Основная функция обучения
async function trainModel() {
    // Токенизация текста
    const tokenizer = tokenizeText(texts);
    const sequences = tokenizer.apply(texts).arraySync();
    
    // Преобразование в тензоры
    const xTrain = tf.tensor2d(sequences);
    const yTrain = tf.tensor1d(labels);
    
    // Создание модели
    const vocabSize = tokenizer.getVocabulary().length;
    const model = createModel(vocabSize);
    
    // Обучение модели
    const history = await model.fit(xTrain, yTrain, {
        epochs: 20,
        batchSize: 2,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Эпоха ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
            }
        }
    });
    
    // Сохранение модели
    await model.save('file://./text-model');
    console.log('Модель сохранена.');
    
    // Пример предсказания
    const testText = "Это хороший продукт";
    const testSeq = tokenizer.apply([testText]).arraySync();
    const prediction = model.predict(tf.tensor2d(testSeq)).dataSync()[0];
    console.log(`Предсказание для "${testText}": ${prediction.toFixed(4)} (${prediction > 0.5 ? 'позитивный' : 'негативный'})`);
}

trainModel().catch(console.error);