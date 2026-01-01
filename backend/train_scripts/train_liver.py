import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load liver data (your file HAS header)
df = pd.read_csv("../../datasets/liver.csv")

# Rename columns to the correct names
df.columns = ["Age","Gender","TB","DB","Alkphos","Sgpt","Sgot","TP","ALB","A/G","Dataset"]

# Convert Dataset column: 1 = disease, 2 = healthy
df["Dataset"] = df["Dataset"].apply(lambda x: 1 if x == 1 else 0)

X = df.drop("Dataset", axis=1)
y = df["Dataset"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train, y_train)

with open("../../ml_models/liver.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("✅ liver.pkl saved successfully!")
