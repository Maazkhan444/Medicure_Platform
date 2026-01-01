import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import requests
import io

# Setup Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "ml_models")
os.makedirs(model_dir, exist_ok=True)

def train_and_save(name, url, selected_columns, target_col, text_cols=[]):
    print(f"\n⏳ Training {name}...")
    try:
        # 1. Download
        s = requests.get(url).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')))
        
        # 2. Select Features
        # We filter the dataframe to keep only columns we want in our UI
        if selected_columns:
            df = df[selected_columns + [target_col]]
            
        # 3. Clean Text Data
        for col in text_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                
        # 4. Clean Missing Values (Fill with 0 or Mean)
        df = df.fillna(0)

        # 5. Split
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Auto-detect target if it's text (like 'M'/'B')
        if y.dtype == 'object':
             y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 6. Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        # 7. Train
        model = RandomForestClassifier(n_estimators=50)
        model.fit(X_train, y_train)

        # 8. Save
        save_path = os.path.join(model_dir, f"{name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump((model, scaler), f)
        print(f"✅ Success! {name}.pkl saved.")
        print(f"   Inputs required: {list(X.columns)}")

    except Exception as e:
        print(f"❌ Failed {name}: {e}")

# --- 1. LIVER DISEASE ---
# Dataset: Indian Liver Patient Records
# We map gender Male/Female to numbers
train_and_save(
    name="liver",
    url="https://raw.githubusercontent.com/dphi-official/Datasets/master/indian_liver_patient.csv",
    selected_columns=["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase", "Alamine_Aminotransferase", "Total_Protiens", "Albumin"],
    target_col="Dataset",
    text_cols=["Gender"]
)

# --- 2. KIDNEY DISEASE ---
# Dataset: Chronic Kidney Disease (Cleaned version)
train_and_save(
    name="kidney",
    url="https://raw.githubusercontent.com/mansoordaku/CKD-Prediction-App/master/kidney_disease.csv",
    selected_columns=["age", "bp", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo"],
    target_col="classification",
    text_cols=[]
)

# --- 3. CANCER ---
# Dataset: Breast Cancer Wisconsin
# We select the "mean" features (first 10) to keep UI simple
train_and_save(
    name="cancer",
    url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/dowjones.csv", # Placeholder check below
    # We use a reliable raw link for cancer
    # url="https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Cancer.csv", 
    # Actually, let's use the standard sklearn one logic but via CSV for consistency
    selected_columns=["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean"],
    target_col="diagnosis",
    text_cols=[]
)

# RE-RUN CANCER WITH CORRECT URL
train_and_save(
    name="cancer",
    url="https://raw.githubusercontent.com/rashida048/Datasets/master/cancer.csv",
    selected_columns=["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"],
    target_col="diagnosis",
    text_cols=[]
)