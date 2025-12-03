from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)
bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
bagging_acc = accuracy_score(y_test, y_pred_bagging)

boosting = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    algorithm="SAMME",
    random_state=42
)
boosting.fit(X_train, y_train)
y_pred_boosting = boosting.predict(X_test)
boosting_acc = accuracy_score(y_test, y_pred_boosting)

print("Bagging Accuracy:", bagging_acc)
print("Boosting Accuracy:", boosting_acc)
print("\n--- Bagging Classification Report---")
print(classification_report(y_test, y_pred_bagging, target_names=iris.target_names))
print("\n--- Boosting Classification Report---")
print(classification_report(y_test, y_pred_boosting, target_names=iris.target_names))
