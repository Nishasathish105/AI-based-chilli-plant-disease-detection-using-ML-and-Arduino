import os
import numpy as np
from feature_extraction import extract_features
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

data = []
labels = []

# 0 = healthy, 1 = diseased
folders = ["healthy", "diseased"]

for label, folder in enumerate(folders):
    path = "dataset/" + folder

    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            features = extract_features(img_path)

            data.append(features)
            labels.append(label)

        except:
            pass

print("Total samples:", len(data))

X = np.array(data)
y = np.array(labels)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# models
models = {
    "SVM": SVC(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier()
}

best_acc = 0
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print(name, "Accuracy:", acc)

    if acc > best_acc:
        best_acc = acc
        best_model = model

# save best model
joblib.dump(best_model, "best_model.pkl")
print("Best model saved with accuracy:", best_acc)
