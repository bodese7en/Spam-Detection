import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load the tokenizer and model
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
model = tf.keras.models.load_model('my_model.keras')  


# Streamlit UI
st.title("Spam Detection App")

# User input text box
user_input = st.text_area("Enter a message:")

# Prediction function
def predict_spam(predict_msg):
    sequence = tokenizer.texts_to_sequences([predict_msg])
    padded = pad_sequences(sequence, padding='post', maxlen=40)  # Adjust maxlen as needed
    prediction = model.predict(padded)
    return "Spam" if prediction[0][0] > 0.5 else "Not Spam"

# Predict and display result when the user clicks a button
if st.button("Predict"):
    if user_input:
        prediction = predict_spam(user_input)
        st.write(f"Prediction: {prediction}")
    else:
        st.warning("Please enter a message.")
