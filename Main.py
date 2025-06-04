import numpy as np
import pickle
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

new_model = True   # Создавать ли новую модель


if new_model:

    # Загрузка данных
    with open("Data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Токенизация
    tokenizer = Tokenizer(char_level=False, filters="")  # можно попробовать char-level
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1

    # Создание последовательностей
    sequences = []
    for line in text.split("\n"):
        if not line:
            continue
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in sequences])
    sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding="pre")

    X = sequences[:, :-1]
    y = sequences[:, -1]

    # Создание модели
    model = Sequential([
        Embedding(total_words, 100, input_length=max_sequence_len-1),
        LSTM(150, return_sequences=True),
        LSTM(100),
        Dense(total_words, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    model.fit(X, y, epochs=100, verbose=1)

    # После обучения модели:
    save_model(model, "rishny_model.h5")  # Сохраняем модель

    with open("tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f)  # Сохраняем токенизатор

    with open("config.pickle", "wb") as f:
        pickle.dump({"max_sequence_len": max_sequence_len}, f)  # Сохраняем настройки

else:

    def load_rishny_ai():
        model = load_model("rishny_model.h5")  # Загружаем модель
    
        with open("tokenizer.pickle", "rb") as f:
            tokenizer = pickle.load(f)  # Загружаем токенизатор
    
        with open("config.pickle", "rb") as f:
            config = pickle.load(f)  # Загружаем настройки
    
        return model, tokenizer, config["max_sequence_len"]

    # Проверяем, есть ли сохранённая модель
    try:
        model, tokenizer, max_sequence_len = load_rishny_ai()
        print("Модель загружена!")
    except:
        print("Модель не найдена. Обучаем с нуля...")
        # Здесь код обучения из предыдущего примера

# Генерация текста
def generate_text(seed_text, next_words=100):
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
    return seed_text

while True:
    print(generate_text(input("Введите фразу: ")))