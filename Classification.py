import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv("reviews.csv")
df = df[['review','sentiment']]
print("Dataset Preview:")
print(df.head())
df['sentiment'] = df['sentiment'].map({'negative':0, 'positive':1})
X = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
msg = ["The product quality is very good and I am satisfied"]
msg_vector = vectorizer.transform(msg)
prediction = model.predict(msg_vector)
if prediction[0] == 1:
    print("\nThis review is POSITIVE")
else:
    print("\nThis review is NEGATIVE")