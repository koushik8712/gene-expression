import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("data/GSE7305_matrix.csv", index_col=0).T
labels = ['healthy'] * 10 + ['disease'] * 10  # manually labeled from NCBI metadata

# Variance-based feature selection
selector = VarianceThreshold(threshold=0.5)
df_selected = selector.fit_transform(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df_selected, labels, test_size=0.3, random_state=42)

# Train RandomForest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, clf.predict(X_test)))

# Save model
joblib.dump(clf, 'model/classifier.pkl')
