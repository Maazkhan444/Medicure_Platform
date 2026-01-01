import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# 1. Setup Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "datasets", "heart.csv")
model_path = os.path.join(script_dir, "ml_models", "heart.pkl")

print(f"🔍 Reading data from: {csv_path}")

# 2. Load the Local File
if not os.path.exists(csv_path):
    print("❌ ERROR: backend/datasets/heart.csv not found.")
    exit()

df = pd.read_csv(csv_path)
print(f"✅ Data Loaded: {len(df)} rows")

# --- 3. AUTO-CLEANING (The Magic Fix) ---
# This finds any column with text (like 'Male') and converts it to numbers
print("🧹 Cleaning data (converting text to numbers)...")

for col in df.select_dtypes(include=['object']).columns:
    print(f"   - Converting column '{col}' to numbers")
    # Convert text to category codes (e.g., Male -> 1, Female -> 0)
    df[col] = df[col].astype('category').cat.codes

# --- 4. Prepare Data ---
target_col = "target"

# Handle different dataset versions
if "target" not in df.columns:
    if "HeartDisease" in df.columns: # Common in some versions
        target_col = "HeartDisease"
    elif "num" in df.columns: # Common in UCI version
        target_col = "num"
    else:
        # Last resort: assume the last column is the target
        target_col = df.columns[-1]
        print(f"⚠️ Warning: 'target' column not found. Using last column: '{target_col}'")

print(f"🎯 Target column is: {target_col}")

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Scale & Train ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("⏳ Training model...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# --- 6. Save ---
with open(model_path, "wb") as f:
    pickle.dump((model, scaler), f)

print(f"🎉 Success! heart.pkl saved to {model_path}")