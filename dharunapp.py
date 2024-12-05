import json
import numpy as np
import pandas as pd
import re
import random
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, LayerNormalization, Dense, Dropout

app = Flask(__name__)

# Load and preprocess the intents data
with open('intents.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['intents'])

# Transform data into a more usable format
dic = {"tag": [], "patterns": [], "responses": []}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)

df = pd.DataFrame.from_dict(dic)

# Data Preprocessing
tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(df['patterns'])
vocab_size = len(tokenizer.word_index)

# Convert patterns to sequences and pad them
ptrn2seq = tokenizer.texts_to_sequences(df['patterns'])
X = pad_sequences(ptrn2seq, padding='post')

# Encode the target labels
lbl_enc = LabelEncoder()
y = lbl_enc.fit_transform(df['tag'])

# Build the LSTM model
model = Sequential()
model.add(Input(shape=(X.shape[1],)))
model.add(Embedding(input_dim=vocab_size + 1, output_dim=100, mask_zero=True))
model.add(LSTM(32, return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(32, return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(32))
model.add(LayerNormalization())
model.add(Dense(128, activation="relu"))
model.add(LayerNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(LayerNormalization())
model.add(Dropout(0.2))
model.add(Dense(len(np.unique(y)), activation="softmax"))
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Train the model (uncomment when running for the first time)
# model.fit(x=X, y=y, batch_size=10, epochs=100, callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)])

def generate_answer(pattern):
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', pattern)
    txt = txt.lower()
    txt = txt.split()
    txt = " ".join(txt)
    text.append(txt)

    x_test = tokenizer.texts_to_sequences(text)
    x_test = np.array(x_test).squeeze()
    x_test = pad_sequences([x_test], padding='post', maxlen=X.shape[1])
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()
    tag = lbl_enc.inverse_transform([y_pred])[0]

    responses = df[df['tag'] == tag]['responses'].values[0]
    return random.choice(responses)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    response = generate_answer(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
