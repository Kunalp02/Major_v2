import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load the saved model
model = tf.keras.models.load_model('text_completion_model.h5')

# Load the tokenizer used for training
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Generate text completion
def generate_text_completion(seed_text, next_words, max_sequence_length):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted_probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Example usage
seed_text = "The quick brown fox"
next_words = 3
max_sequence_length = 5  # Define this based on your training data
completed_text = generate_text_completion(seed_text, next_words, max_sequence_length)
print(completed_text)
