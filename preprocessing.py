import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("fitness.csv")
data.columns = data.columns.str.strip()
data = data.fillna(data.mean(numeric_only=True))
print(data.head())
x = data[['Workout Hours', 'Calories Intake', 'Sleep Hours']]
y_reg = data['Weight Loss']
y_clf = (data['Weight Loss'] >= 3).astype(int)
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_train, x_test, y_reg_train, y_reg_test = train_test_split(
    x, y_reg, test_size=0.2, random_state=42
)
_, _, y_clf_train, y_clf_test = train_test_split(
    x, y_clf, test_size=0.2, random_state=42
)
lin_model = LinearRegression()
lin_model.fit(x_train, y_reg_train)
y_reg_pred = lin_model.predict(x_test)
print("\nLinear Regression Predictions:")
print(y_reg_pred[:5])
mse_reg = mean_squared_error(y_reg_test, y_reg_pred)
r2_reg = r2_score(y_reg_test, y_reg_pred)
print("Linear Regression MSE:", mse_reg)
print("Linear Regression R2 Score:", r2_reg)
log_model = LogisticRegression(max_iter=200)
log_model.fit(x_train, y_clf_train)
y_clf_pred = log_model.predict(x_test)
print("\nLogistic Regression Predictions:")
print(y_clf_pred[:5])
print("Accuracy:", accuracy_score(y_clf_test, y_clf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_clf_test, y_clf_pred))
print("Classification Report:\n", classification_report(y_clf_test, y_clf_pred))
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_clf_train)
y_knn_pred = knn.predict(x_test)
print("\nKNN Predictions:")
print(y_knn_pred[:5])
print("KNN Accuracy:", accuracy_score(y_clf_test, y_knn_pred))
print("KNN Confusion Matrix:\n", confusion_matrix(y_clf_test, y_knn_pred))
print("KNN Classification Report:\n", classification_report(y_clf_test, y_knn_pred))
cv_scores = cross_val_score(log_model, x, y_clf, cv=5)
print("\nCross Validation Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())