import streamlit as st
@st.cache_resource 
def install_joblib():
    !pip install joblib==1.3.2
        
    install_joblib()
    import joblib

import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import torch


rf_model = joblib.load('rf_model.pkl')  # Assurez-vous que 'rf_model.pkl' est dans le même répertoire

model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

def encode_tweet(tweet):
    inputs = tokenizer(tweet, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

st.title("Détection de tweets suspects")
tweet_input = st.text_input("Entrez un tweet :")

if st.button("Prédire"):

    embedding = encode_tweet(tweet_input)

    prediction = rf_model.predict(np.array([embedding]))[0]


    if prediction == 1:
        st.write("Ce tweet est suspect.")
    else:
        st.write("Ce tweet n'est pas suspect.")