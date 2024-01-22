from imports import *

path = "https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection"
data = pd.read_csv(path, names = ["Flag", "text"], sep='\t')

preprocess(data)

data["LEN"] = data["text"].str.split(r"(?!^)").str.len()
data["NB"] = data["text"].str.split().str.len()

preparation = ColumnTransformer(
    transformers=[
        ('data_cat', CountVectorizer(), "text"),
        ('data_num', RobustScaler(), ["LEN","NB"])
])

pipel = Pipeline(steps=[
                        ('vector/Scale', preparation),
                        ('modelisation', DecisionTreeClassifier())
                        ])

# Features
X = data.drop("Flag", axis = 1, inplace=False)
# Target
y = data['Flag']

#Z = input()
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

pipel.fit(X_train,y_train)
y_pred = pipel.predict(X_test)
print("Prediction with Count Vectorizer et Logistic Regression:", recall_score(y_test, y_pred, pos_label='spam'))

def find(x):
    if x == 1:
        print ("Message is SPAM")
    else:
        print ("Message is NOT Spam")




#print("Prediction with Count Vectorizer et Logistic Regression:", recall_score(y_test, y_pred, pos_label='spam'))

#print(confusion_matrix(y_test, y_pred))

#lb_encod = LabelEncoder()
#cm_plot = ConfusionMatrixDisplay(confusion_matrix,
#                                display_labels=np.unique(lb_encod.inverse_transform(model.classes_)))

#cm_plot.plot()
"""
TP = sum((y_test == 'spam') & (y_pred == 'spam'))
print(TP)

FN = sum((y_test == 'spam') & (y_pred == 'ham'))
print(FN)

print(TP / (TP + FN))"""


# Test TFIDF et KNN | TFIDF et Logistic regression | CountVectorizer et KNN. Moins précis
"""X = vectorizertf.fit_transform(corpus)
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

X = vectorizerCV.fit_transform(corpus)
y = data['flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 3)
model = KNeighborsClassifier(3)
model.fit(X_train,y_train)
y_predCV = model.predict(X_test)
print("MAE with Count Vectorizer et KNN:", accuracy_score(y_test, y_predCV))"""