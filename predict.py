import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the saved model and Tokenizer instance
loaded_model = tf.keras.models.load_model('my_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Recreate tokenization and padding
full_text = open('text_to_predict.txt', encoding = 'iso-8859-1').read()
full_text = full_text.encode("latin-1").decode("utf-8")
split_text = full_text.split('\n')
texts = []
for i in split_text:
  if len(i) > 20:
      texts.append(i)
      
new_sequences = tokenizer.texts_to_sequences(texts)
max_sequence_length_train = loaded_model.input_shape[1]  # get the original max sequence length from the model input shape
new_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length_train, padding='post')

# Use the loaded model to make predictions
predictions = loaded_model.predict(new_sequences)

predictionsum = 0

for i in range(len(texts)):
  predictionsum += predictions[i][0]

print(predictionsum/len(texts))
