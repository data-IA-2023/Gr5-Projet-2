from imports import *

path = "https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection"
data = pd.read_csv(path, names = ["flag", "text"], sep='\t')
pipe = data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

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