import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# App title
st.title("Fever Detection App")
st.write("This app predicts if a patient has a fever based on symptoms provided.")

# Load the pretrained model and tokenizer
@st.cache_resource  # Cache the model and tokenizer to avoid reloading
def load_model():
    model = BertForSequenceClassification.from_pretrained("fever_detection_model")
    tokenizer = BertTokenizer.from_pretrained("fever_detection_tokenizer")
    return model, tokenizer

model, tokenizer = load_model()

# Prediction function
def predict(text):
    # Preprocess the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    
    # Generate predictions
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze()
    
    # Extract predictions and probabilities
    predicted_label = torch.argmax(probabilities).item()
    probability_no_fever = probabilities[0].item()
    probability_fever = probabilities[1].item()

    # Define diagnosis and recommended medication
    diagnosis = "Fever detected" if predicted_label == 1 else "No fever"
    medication = {
        "Fever detected": "Paracetamol",
        "No fever": "Rest and hydration"
    }

    return diagnosis, medication[diagnosis], probability_no_fever, probability_fever

# User input section
st.subheader("Enter Symptoms")
input_text = st.text_area("Describe the symptoms below:")

if st.button("Predict"):
    if input_text.strip():  # Ensure input is not empty
        # Get prediction
        diagnosis, recommended_medication, prob_no_fever, prob_fever = predict(input_text)

        # Display results
        st.write(f"### Prediction: {diagnosis}")
        st.write(f"**Recommended Medication**: {recommended_medication}")
        st.write(f"**Confidence Levels**:")
        st.write(f"- No Fever: {prob_no_fever:.2f}")
        st.write(f"- Fever: {prob_fever:.2f}")
    else:
        st.warning("Please enter a description of the symptoms.")
