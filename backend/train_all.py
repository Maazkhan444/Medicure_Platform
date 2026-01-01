import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import requests
import io

# 1. Setup Folders
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "ml_models")
os.makedirs(model_dir, exist_ok=True)

print(f"📂 Models will be saved to: {model_dir}")

# --- HELPER FUNCTION: Train a Model ---
def train_model(name, url, columns, target_col):
    print(f"\n⏳ Processing {name}...")
    
    try:
        # Download Data
        s = requests.get(url).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')))
        
        # If dataset has no headers (like Diabetes), assign them
        if columns and len(df.columns) == len(columns):
            df.columns = columns
            
        print(f"   Data Loaded: {len(df)} rows")

        # Prepare X and y
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        model = RandomForestClassifier(n_estimators=50)
        model.fit(X_train, y_train)

        # Save
        save_path = os.path.join(model_dir, f"{name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump((model, scaler), f)
            
        print(f"✅ Success! {name}.pkl saved.")
        
    except Exception as e:
        print(f"❌ Failed to train {name}: {e}")

# --- 2. TRAIN HEART (Clean Dataset) ---
train_model(
    name="heart", 
    url="https://raw.githubusercontent.com/rishabh89007/Heart_Disease_Prediction/master/heart.csv",
    columns=None, # Has headers
    target_col="target"
)

# --- 3. TRAIN DIABETES (Pima Indians Dataset) ---
# Diabetes CSV usually has NO headers, so we define them
diabetes_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
train_model(
    name="diabetes",
    url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    columns=diabetes_cols,
    target_col="Outcome"
)