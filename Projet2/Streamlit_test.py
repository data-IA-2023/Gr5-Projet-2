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
    slide_radio = st.slider(
        "Choisissez le nombre des messages affichés",
        min_value = 0, max_value = len(datasl[datasl["flag"] == add_radio])
    )
if add_radio == "spam":
    

corpus = data["text"]
vectorizertf = TfidfVectorizer()
vectorizerCV = CountVectorizer()


X = vectorizerCV.fit_transform(corpus)
y = data['flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 3)
model = KNeighborsClassifier(3)
model.fit(X_train,y_train)
y_predCV = model.predict(X_test)
print("MAE with Count Vectorizer et KNN:", accuracy_score(y_test, y_predCV))

X = vectorizerCV.fit_transform(corpus)
y = data['flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 3)
model = LogisticRegression(random_state=5)
model.fit(X_train,y_train)
y_predCV = model.predict(X_test)
print("MAE with Count Vectorizer et Logistic Regression:", accuracy_score(y_test, y_predCV))

X = vectorizertf.fit_transform(corpus)
y = data['flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 3)
model = KNeighborsClassifier(3)
model.fit(X_train,y_train)
y_predtf = model.predict(X_test)
print("MAE with TF-IDF et KNN:", accuracy_score(y_test, y_predtf))

X = vectorizertf.fit_transform(corpus)
y = data['flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 3)
model = LogisticRegression(random_state=5)
model.fit(X_train,y_train)
y_predtf = model.predict(X_test)
print("MAE with TF-IDF Logistic Regression:", accuracy_score(y_test, y_predtf))
