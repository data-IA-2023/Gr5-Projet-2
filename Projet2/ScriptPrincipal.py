from imports import *

# Importation CSV
path = "https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection"
data = pd.read_csv(path, names = ["Flag", "text"], sep='\t')

# Preprocessing, voir fonction dans "imports"
preprocess(data)

# Création de deux nouvelles colonnes, longueur des phrases et nombre de mots
data["LEN"] = data["text"].str.split(r"(?!^)").str.len()
data["NB"] = data["text"].str.split().str.len()

sm = SMOTENC(categorical_features = ['text'], random_state= 3)

# Features
X = data.drop("Flag", axis = 1, inplace=False)
# Target
y = data['Flag']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Variables resamples pour Smote
X_smoted, y_smoted = sm.fit_resample(X_train, y_train)

# Transformer
preparation = ColumnTransformer(
    transformers=[
        ('data_cat', CountVectorizer(), "text"),
        ('data_num', RobustScaler(), ["LEN","NB"])
])

# Pipeline
pipel = Pipeline(steps=[
                        ('vector/Scale', preparation),
                        ('modelisation', DecisionTreeClassifier())
                        ])

# Dictionnaire pour GridSearch                        
params = {
    'modelisation': [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier()],
}


scorer = {'accuracy': 'accuracy', 'recall': make_scorer(recall_score, pos_label='spam')}
searchCV = GridSearchCV(pipel, params, scoring=scorer, refit='recall')


searchCV.fit(X_smoted,y_smoted)

st.write('Explorations de données')

with st.sidebar:
    add_radio = st.radio(
        "Ham ou Spam ?",
        ("spam", "ham")
    )
    charting = st.radio(
        "Longueur des messages ou longueur des caractères ?",
        ("LEN", "NB")
    )
    nom_mess = st.slider(
        "Choisir le nombre de messages affichés",
        min_value = 0, max_value = len(data[data["Flag"] == add_radio])

    )

# Exploration de données

datafilter = data[data["Flag"] == add_radio].head(nom_mess)
st.dataframe(datafilter)
st.bar_chart(datafilter[charting])

st.write("Pour la prédiction d'un nouveau message")
txt = st.text_input('Veuillez entrer votre message', 'mail')
lennn = len(re.split((r"(?!^)"), txt))
nomnom = len(re.split(" ", txt))
newmail  = pd.DataFrame.from_dict({"text": [txt], "LEN": [lennn], "NB": [nomnom]})
bestscore = round(searchCV.best_score_,3)
st.write(f"Le modèle de machine learning possède {bestscore} % de précision")

if st.button("Apuyez ici pour prédir avec notre modèle si le message reçu est un spam ou un ham (message normal)"):
    st.write(f"Le mail est un: {searchCV.predict(newmail)[0]}")