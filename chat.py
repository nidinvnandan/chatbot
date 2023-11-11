import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

import random
import spacy
import json
import pickle
from sklearn.metrics import precision_score

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    st.write("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

print("Processing the Intents.....")
with open('intents.json') as json_data:
    intents = json.load(json_data)

print("Looping through the Intents to Convert them to words, classes, documents.......")
words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence using spaCy
        w = [token.text.lower() for token in nlp(pattern)]
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print("Stemming, Lowering, and Removing Duplicates.......")
words = list(set(words))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique words", words)

print("Creating the Data for our Model.....")
training = []
output = []
output_empty = [0] * len(classes)

print("Creating Training Set, Bag of Words for our Model....")
for doc in documents:
    bag = [1 if w in doc[0] else 0 for w in words]

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

print("Shuffling Randomly and Converting into Numpy Array for Faster Processing......")
random.shuffle(training)

bags, output_rows = zip(*training)

train_x = np.array(bags)
train_y = np.array(output_rows)

print("Creating Train and Test Lists.....")
train_x = list(train_x)
train_y = list(train_y)

print("Building Neural Network for Our Chatbot to be Contextual....")
print("Resetting graph data....")
model = Sequential()
model.add(Dense(8, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)

print("Training....")
model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=8, verbose=1, callbacks=[early_stopping])

print("Saving the Model.......")
model.save('model_keras.h5')

print("Pickle is also Saved..........")
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open("training_data", "wb"))

print("Loading Pickle.....")
data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

print("Loading the Model......")
model = tf.keras.models.load_model('model_keras.h5')

# Evaluate the model
print("Evaluating the Model....")
loss, accuracy = model.evaluate(np.array(train_x), np.array(train_y))
print(f"Training Loss: {loss}")
print(f"Training Accuracy: {accuracy}")

ERROR_THRESHOLD = 0.25
print("ERROR_THRESHOLD = 0.25")


def clean_up_sentence(sentence):
    sentence_words = [token.text.lower() for token in nlp(sentence)]
    return sentence_words


def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)


def classify(sentence):
    results = model.predict(np.array([bow(sentence, words)]))[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [(classes[r[0]], r[1]) for r in results]
    return return_list


def response(sentence, userID='123', show_details=False):
    if sentence.lower() in ['quit', 'bye']:
        return "Goodbye! Chatbot session terminated."
    results = classify(sentence)
    if results:
        for i in intents['intents']:
            if i['tag'] == results[0][0]:
                return random.choice(i['responses'])

#run the model in the main.py file
