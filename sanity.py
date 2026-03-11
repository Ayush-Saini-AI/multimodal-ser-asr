import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load your generated real dataset
df = pd.read_csv("real_features.csv")
X = df[["sentiment_score", "pitch_mean", "energy_mean", "tempo"]].values
y = df["emotion"].values

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluate and print the report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))