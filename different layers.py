import pandas as pd
import numpy as np

train_data = pd.read_csv("train_E6oV3lV.csv")
test_data = pd.read_csv("test_tweets_anuFYb8.csv")

sentences = []
labels = []

for i in range(0, len(train_data)):
    sentences.append(train_data['tweet'][i])
    labels.append(train_data['label'][i])
    
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

final_sentences = []
signs = [".", "!", "@", "#", "$", "%", "^", "&", "*", ",", ".", "...", ";", "-"]

for line in sentences:
    sentence = []
    for word in line.split():
        if word not in stop_words:
            if word not in signs:
                sentence.append(word)
    final_sentences.append(sentence)

tokenizer = Tokenizer(oov_token = "<oov>", num_words = 20000)
tokenizer.fit_on_texts(final_sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(final_sentences)
padded = pad_sequences(sequences, padding = 'post', truncating = 'post', maxlen = 27)

sentences = []
for i in range(0, len(test_data)):
    sentences.append(test_data['tweet'][i])

final_test_sentences = []
for line in sentences:
    sentence = []
    for word in line.split():
        if word not in stop_words:
            if word not in signs:
                sentence.append(word)
    final_test_sentences.append(sentence)

test_sequences = tokenizer.texts_to_sequences(final_test_sentences)
test_padded = pad_sequences(test_sequences, padding = 'post', truncating = 'post', maxlen = 27)

import tensorflow as tf

train_set = padded[:25000, :]
valid_set = padded[25000:, :]
train_labels = np.array(labels[:25000])
valid_labels = np.array(labels[25000:])
labels = np.array(labels)

# Model 1 using Flatten layer
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(20000, 32, input_length = 27),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

num_epochs = 10
model.fit(train_set, train_labels, epochs = num_epochs, validation_data = (valid_set, valid_labels))

model.fit(padded, labels, epochs = 5)
test_labels = model.predict(test_padded)

for i in range(0, len(test_labels)):
    if(test_labels[i] < 0.5):
        test_labels[i] = 0
    else:
        test_labels[i] = 1

test_data['label'] = test_labels
test_data.label = test_data.label.astype(int)
test_data_new = test_data[['id', 'label']]

test_data_new.to_csv("Solution.csv", index = False)

# Model 2 using GRU layer
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(20000, 32, input_length = 27),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

num_epochs = 10
model.fit(train_set, train_labels, epochs = num_epochs, validation_data = (valid_set, valid_labels))

model.fit(padded, labels, epochs = 5)
test_labels = model.predict(test_padded)

for i in range(0, len(test_labels)):
    if(test_labels[i] < 0.5):
        test_labels[i] = 0
    else:
        test_labels[i] = 1

test_data['label'] = test_labels
test_data.label = test_data.label.astype(int)
test_data_new = test_data[['id', 'label']]

test_data_new.to_csv("Solution_2.csv", index = False)

# Model 3 using LSTM layer
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(20000, 32, input_length = 27),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

num_epochs = 10
model.fit(train_set, train_labels, epochs = num_epochs, validation_data = (valid_set, valid_labels))

model.fit(padded, labels, epochs = 3)
test_labels = model.predict(test_padded)

for i in range(0, len(test_labels)):
    if(test_labels[i] < 0.5):
        test_labels[i] = 0
    else:
        test_labels[i] = 1

test_data['label'] = test_labels
test_data.label = test_data.label.astype(int)
test_data_new = test_data[['id', 'label']]

test_data_new.to_csv("Solution_3.csv", index = False)

# Model 4 using convollutional layer
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(20000, 32, input_length = 27),
    tf.keras.layers.Conv1D(128, 5, activation = "relu"),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

num_epochs = 10
model.fit(train_set, train_labels, epochs = num_epochs, validation_data = (valid_set, valid_labels))

model.fit(padded, labels, epochs = 3)
test_labels = model.predict(test_padded)

for i in range(0, len(test_labels)):
    if(test_labels[i] < 0.5):
        test_labels[i] = 0
    else:
        test_labels[i] = 1

test_data['label'] = test_labels
test_data.label = test_data.label.astype(int)
test_data_new = test_data[['id', 'label']]

test_data_new.to_csv("Solution_4.csv", index = False)
