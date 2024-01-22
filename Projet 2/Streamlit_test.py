import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

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

# preprocessing
path = "https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection"
data = pd.read_csv(path, names = ["flag", "text"], sep='\t')
data["len"] = data["text"].str.split(r"(?!^)").str.len()
data["nb"] = data["text"].str.split().str.len()
data.dropna(inplace= True)
data.drop_duplicates(inplace = True)

# agroupement de encodage et modelisation

preparation = ColumnTransformer(
    transformers=[
        ('data_cat', CountVectorizer() , "text"),
        ('data_num', RobustScaler(), ["len","nb"])
    ])

pipe = Pipeline(steps=[
                       ('preparation', preparation),
                       ('modelisation', DecisionTreeClassifier())
                       ])

# diviser train et test, appel du pipeline et regarder la performance

X = data.drop(["flag"], axis = 1)
y = data["flag"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 3)
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
print("Score avec tree classifier et count vectorizer", recall_score(y_test, y_pred, pos_label="spam"))

