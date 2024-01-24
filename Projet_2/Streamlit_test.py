import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import make_scorer
from imblearn.over_sampling import SMOTENC
import streamlit as st

# preprocessing
path = "https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection"
data = pd.read_csv(path, names = ["flag", "text"], sep='\t')
data["len"] = data["text"].str.split(r"(?!^)").str.len()
data["nb"] = data["text"].str.split().str.len()
data.dropna(inplace= True)
data.drop_duplicates(inplace = True)
sm = SMOTENC(categorical_features = ["text"], random_state= 3)

# agroupement de encodage et modelisation

preparation = ColumnTransformer(
    transformers=[
        ('data_cat', CountVectorizer() , "text"),
        ('data_num', RobustScaler(), ["len","nb"])
    ])

model = Pipeline(steps=[
                       ('preparation', preparation),
                       ('modelisation', LogisticRegression())
                       ])


params = {
        'modelisation': [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier()]
}

X = data.drop(["flag"], axis = 1)
y = data["flag"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 3, stratify=y)

X_smot, y_smot = sm.fit_resample(X_train, y_train)

scorers = {"accuracy": "accuracy", "recall": make_scorer(recall_score, pos_label = "spam")}

grid = GridSearchCV(model, params, scoring=scorers, refit = "recall")

# diviser train et test, appel du pipeline et regarder la performance

grid.fit(X_smot,y_smot)

with st.sidebar:
    add_radio = st.radio(
        "Choisissez ham ou spam",
        ("spam", "ham")
    )
    nom_mess = st.slider(
        "Choisissez le nombre des messages affichés",
        min_value = 0, max_value = len(data[data["flag"] == add_radio])
    )

st.write(data[data["flag"] == add_radio].head(nom_mess))
txt = st.text_input('Introduisez un mail', 'mail')
leng = len(re.split((r"(?!^)"), txt))
nom = len(re.split(" ", txt))
newmail  = pd.DataFrame.from_dict({"text": [txt], "len": [leng], "nb": [nom]})
bestscore = round(grid.best_score_,3)
st.write(f"Le model de Machine Learning est trainé avec un score de {bestscore} % de precision")

if st.button("Apuyez ici pour prédir avec notre modèle si le mail que vous avez ecrit est un spam ou un veritable mail(ham)"):
    st.write(f"Le mail est un: {grid.predict(newmail)[0]}")
