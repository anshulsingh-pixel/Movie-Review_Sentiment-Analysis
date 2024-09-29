import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load your trained model
model = load_model('new.h5')

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=250)
    return padded_review

# Title and Introduction
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review, and the model will predict whether the review is **positive** or **negative**.")

# Text input for the user to provide a movie review
review = st.text_area("Enter your movie review:")

# Predict button
if st.button("Analyze"):
    if review:
        # Preprocess the input
        preprocessed_input=preprocess_text(review)

        # Make a prediction
        prediction = model.predict(preprocessed_input)

        # Determine if it's positive or negative
        sentiment = 'Positive' if prediction[0][0] > 0.05 else 'Negative'

        # Display the result
        st.write(f"The sentiment of the review is **{sentiment}**.")
        #st.write(f'Prediction: {prediction[0][0]}.')
    else:
        st.write("Please enter a review.")
