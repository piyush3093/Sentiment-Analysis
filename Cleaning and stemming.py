''' Text Cleaning '''

import pandas as pd
import numpy as np

train_data = pd.read_csv("train_E6oV3lV.csv")
test_data = pd.read_csv("test_tweets_anuFYb8.csv")

train_labels = train_data["label"]

full_data = pd.concat([train_data, test_data], axis = 0)

full_data["tidy_tweet"] = np.nan
full_data = full_data.reset_index()
full_data = full_data.drop('index', axis = 1)

for i in range(0, len(full_data)):
    full_data["tidy_tweet"][i] = full_data.tweet[i].replace("@user ", "")
full_data["tidy_tweet"] = full_data["tidy_tweet"].str.replace("[^a-zA-Z#]", " ")
full_data["tidy_tweet"] = full_data["tidy_tweet"].str.lower()

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

final_sentences = []

for line in full_data["tidy_tweet"]:
    sentence = []
    for word in line.split():
        if word not in stop_words:
            sentence.append(word)
    final_sentences.append(sentence)

train_sentences = final_sentences[:len(train_data)]
test_sentences = final_sentences[len(train_data):]

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(oov_token = "<oov>", num_words = 20000)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(train_sentences)
max_length = max([len(x) for x in sequences])
padded = pad_sequences(sequences, maxlen = 25, padding = 'post', truncating = 'post')

test_sequences = tokenizer.texts_to_sequences(test_sentences)
pad_test = pad_sequences(test_sequences, maxlen = 25, padding = 'post', truncating = 'post')

# Defining the train and valid set

labels = [x for x in train_labels]
train_set = padded[:25000]
valid_set = padded[25000:]
train_labels = np.array(labels[:25000])
valid_labels = np.array(labels[25000:])
labels = np.array(labels)

# Using LSTM layer
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(20000, 32, input_length = 25),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

num_epochs = 10
model.fit(train_set, train_labels, epochs = num_epochs, validation_data = (valid_set, valid_labels))

model.fit(padded, labels, epochs = 5)
test_labels = model.predict(pad_test)

for i in range(0, len(test_labels)):
    if(test_labels[i] < 0.5):
        test_labels[i] = 0
    else:
        test_labels[i] = 1

test_data['label'] = test_labels
test_data.label = test_data.label.astype(int)
test_data_new = test_data[['id', 'label']]

test_data_new.to_csv("Solution_5.csv", index = False)

# Using Stemming
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

final_sentences_new = []
for line in final_sentences:
    sentence = []
    for word in line:
        sentence.append(stemmer.stem(word))
    final_sentences_new.append(sentence)

train_sentences = final_sentences[:len(train_data)]
test_sentences = final_sentences[len(train_data):]

tokenizer = Tokenizer(oov_token = "<oov>", num_words = 20000)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(train_sentences)
max_length = max([len(x) for x in sequences])
padded = pad_sequences(sequences, maxlen = 25, padding = 'post', truncating = 'post')

test_sequences = tokenizer.texts_to_sequences(test_sentences)
pad_test = pad_sequences(test_sequences, maxlen = 25, padding = 'post', truncating = 'post')

train_set = padded[:25000]
valid_set = padded[25000:]

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(20000, 32, input_length = 25),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

num_epochs = 10
model.fit(train_set, train_labels, epochs = num_epochs, validation_data = (valid_set, valid_labels))

model.fit(padded, labels, epochs = 5)
test_labels = model.predict(pad_test)

for i in range(0, len(test_labels)):
    if(test_labels[i] < 0.5):
        test_labels[i] = 0
    else:
        test_labels[i] = 1

test_data['label'] = test_labels
test_data.label = test_data.label.astype(int)
test_data_new = test_data[['id', 'label']]

test_data_new.to_csv("Solution_6.csv", index = False)

all_text = ""

for line in final_sentences_new:
    sentence = " ".join([word for word in line])
    all_text = all_text + " " + sentence

import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordcloud = WordCloud(width = 800, height = 500, random_state = 21, max_font_size = 110).generate(all_text)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()








