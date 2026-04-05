import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("result.csv")
print("Dataset Preview:")
print(df.head())
le = LabelEncoder()
df['result'] = le.fit_transform(df['result'])
X = df.drop("result", axis=1)
y = df["result"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nR2 Score:")
print(r2_score(y_test, y_pred))
scores = cross_val_score(knn, X, y, cv=5)
print("\nCross Validation Scores:", scores)
print("Average Cross Validation Score:", scores.mean())