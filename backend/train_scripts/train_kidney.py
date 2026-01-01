import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("../../datasets/kidney.csv")

# Convert classification column to 0/1
df["classification"] = df["classification"].map({"ckd": 1, "notckd": 0})

# Label encode ALL object columns
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

X = df.drop("classification", axis=1)
y = df["classification"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train, y_train)

with open("../../ml_models/kidney.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("✅ kidney.pkl saved successfully!")
