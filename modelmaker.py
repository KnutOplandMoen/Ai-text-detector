import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle 

#reading traning data and splitting the text and the labels into two seperate lists 
df = pd.read_excel('your_file_name.xlsx')
texts = df.iloc[:,0].tolist()
labels = df.iloc[:,1].tolist()

texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer_train = Tokenizer()
tokenizer_train.fit_on_texts(texts_train)
word_index_train = tokenizer_train.word_index
total_words_train = len(word_index_train) + 1

sequences_train = tokenizer_train.texts_to_sequences(texts_train)
sequences_val = tokenizer_train.texts_to_sequences(texts_val)

max_sequence_length_train = max([len(seq) for seq in sequences_train])
max_sequence_length_val = max([len(seq) for seq in sequences_val])

sequences_train = pad_sequences(sequences_train, maxlen=max_sequence_length_train, padding='post')
sequences_val = pad_sequences(sequences_val, maxlen=max_sequence_length_train, padding='post')  # pad to the same length as training data

labels_train = tf.constant(labels_train)
labels_val = tf.constant(labels_val)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words_train, 16, input_length=max_sequence_length_train),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(3, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00045), loss='binary_crossentropy', metrics=['accuracy'])

epochs = 30 #number of epochs, test this out to see what works the best with your data

es = EarlyStopping(monitor='val_loss', mode='min', baseline=0.2)

history = model.fit(sequences_train, labels_train, epochs=epochs, batch_size=1, 
                    validation_data=(sequences_val, labels_val))

#saving the model for future use
model.save('my_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

#plotting training and validation data
plt.subplot(121)
plt.plot(range(0, epochs), history.history['val_accuracy'], label = 'val_accuracy')
plt.plot(range(0, epochs), history.history['val_loss'], label = 'val_loss')
plt.title('validation data')
plt.legend()

plt.subplot(122)
plt.title('training data')
plt.plot(range(0, epochs), history.history['accuracy'], label = 'training_accuracy')
plt.plot(range(0, epochs), history.history['loss'], label = 'training_loss')
plt.legend()

