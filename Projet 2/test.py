import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

#preprocessing
path = "https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection"
data = pd.read_csv(path, names = ["flag", "text"], sep='\t')
data.dropna(inplace=True)
data.drop_duplicates(inplace = True)

#agroupement de encodage et modelisation
pipe = Pipeline(steps=[
                       ('vectorization', CountVectorizer()),
                       ('modelisation', LogisticRegression())
                       ])
X =data["text"] #corpus
y = data["flag"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 3)
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
print("Presiction avec Count Vectorizer et Logistic Regression:", accuracy_score(y_test, y_pred))