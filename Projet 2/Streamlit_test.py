import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression

path = "https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection"
data = pd.read_csv(path, names = ["flag", "text"], sep='\t')
pipe = data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
datasl = data

datasl["longeur du document"] = datasl["text"].str.split(r"(?!^)").str.len()
datasl["nombre des mots"] = datasl["text"].str.split().str.len()

with st.sidebar:
    add_radio = st.radio(
        "Choisissez ham ou spam",
        ("spam", "ham")
    )
    nom_mess = st.slider(
        "Choisissez le nombre des messages affichés",
        min_value = 0, max_value = len(datasl[datasl["flag"] == add_radio])
    )

st.write(datasl[datasl["flag"] == add_radio].head(nom_mess))
