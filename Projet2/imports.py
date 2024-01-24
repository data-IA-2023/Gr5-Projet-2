import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, make_scorer, accuracy_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTENC

def preprocess(a):
    a.drop_duplicates(inplace=True)
    a.dropna(inplace=True)