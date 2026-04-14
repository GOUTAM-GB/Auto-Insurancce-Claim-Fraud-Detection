import os
import pickle
import pandas as pd
import numpy as np
import warnings
import io
import base64
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb

warnings.filterwarnings('ignore')

# Constants
MODEL_PATH = r'models\fraud_detection_model_max.pkl'
DATA_PATH = r'media\insurance fraud claims.csv'

def load_data(file_path):
    """Load the CSV file"""
    return pd.read_csv(file_path)

def extract_date_features(df, col_name):
    """Extract features from date column"""
    if col_name not in df.columns:
        return df
        
    df[col_name] = pd.to_datetime(df[col_name], dayfirst=True, errors='coerce')
    mode_val = df[col_name].mode()
    fill_val = mode_val[0] if not mode_val.empty else pd.Timestamp('2015-01-01')
    df[col_name] = df[col_name].fillna(fill_val)
    
    df[f'{col_name}_year'] = df[col_name].dt.year
    df[f'{col_name}_month'] = df[col_name].dt.month
    df[f'{col_name}_day'] = df[col_name].dt.day
    df[f'{col_name}_dayofweek'] = df[col_name].dt.dayofweek
    return df.drop(columns=[col_name])

def preprocess_data_max(data, fit=True, saved_artifacts=None):
    """
    Maximum accuracy preprocessing:
    - Uses all features
    - Specialized date extraction
    - Robust categorical encoding
    """
    df = data.copy()
    
    # Handle missing values '?'
    df = df.replace('?', 'Unknown')
    
    # Target variable
    y = None
    if 'fraud_reported' in df.columns:
        y = df['fraud_reported'].apply(lambda x: 1 if x == 'Y' else 0)
        df = df.drop(columns=['fraud_reported'])
        
    # Extract date features
    df = extract_date_features(df, 'policy_bind_date')
    df = extract_date_features(df, 'incident_date')
    
    if fit:
        # Identify column types
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df, y, encoders, scaler, categorical_cols, numeric_cols
    else:
        encoders = saved_artifacts['encoders']
        scaler = saved_artifacts['scaler']
        categorical_cols = saved_artifacts['categorical_cols']
        numeric_cols = saved_artifacts['numeric_cols']
        
        # Ensure input has all numeric columns and they are numeric
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0
                
        for col in categorical_cols:
            if col in df.columns:
                le = encoders[col]
                classes = set(le.classes_)
                df[col] = df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in classes else 0)
            else:
                df[col] = 0
                
        # Scale only numeric columns
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        return df, y

def train_max_accuracy():
    """Train the high-capacity model and save it"""
    os.makedirs('models', exist_ok=True)
    
    data = load_data(DATA_PATH)
    print(f"Data loaded: {data.shape}")
    
    X, y, encoders, scaler, cat_cols, num_cols = preprocess_data_max(data, fit=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    print("Training High-Capacity XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=12,
        learning_rate=0.01,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    y_full_pred = model.predict(X)
    full_acc = accuracy_score(y, y_full_pred)
    
    # Save the model and preprocessing tools
    artifacts = {
        'model': model,
        'encoders': encoders,
        'scaler': scaler,
        'categorical_cols': cat_cols,
        'numeric_cols': num_cols,
        'feature_names': X.columns.tolist()
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(artifacts, f)
    
    print(f"Model trained. Full Dataset Accuracy: {full_acc:.4f}")
    return artifacts

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image_base64

def create_visuals(data, y, y_pred):
    """Create visualization images for the model"""
    fig1 = plt.figure(figsize=(10, 6))
    fraud_counts = pd.Series(y).value_counts()
    colors = ['#2ecc71', '#e74c3c']
    fraud_counts.plot(kind='bar', color=colors, ax=fig1.gca())
    plt.title('Fraud vs Normal Claims (Actual)', fontsize=14, fontweight='bold')
    plt.xticks([0, 1], ['Normal', 'Fraud'], rotation=0)
    barplot_image = fig_to_base64(fig1)
    
    fig2 = plt.figure(figsize=(8, 6))
    conf_matrix = confusion_matrix(y, y_pred)
    sns.heatmap(conf_matrix, annot=True, cmap='Reds', fmt='d',
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.title('Confusion Matrix (Max Accuracy Model)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    confusion_image = fig_to_base64(fig2)
    
    return barplot_image, confusion_image

def main():
    """Main training and evaluation function for the web app"""
    if not os.path.exists(MODEL_PATH):
        artifacts = train_max_accuracy()
    else:
        with open(MODEL_PATH, 'rb') as f:
            artifacts = pickle.load(f)
            
    data = load_data(DATA_PATH)
    X, y = preprocess_data_max(data, fit=False, saved_artifacts=artifacts)
    y_pred = artifacts['model'].predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    
    barplot_image, confusion_image = create_visuals(data, y, y_pred)
    
    return accuracy, precision, recall, barplot_image, confusion_image

def prediction_value(input_data):
    """Predict fraud for a single input record"""
    if not os.path.exists(MODEL_PATH):
        train_max_accuracy()
        
    with open(MODEL_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    
    input_df = pd.DataFrame([input_data])
    
    training_data = load_data(DATA_PATH)
    for col in training_data.columns:
        if col != 'fraud_reported' and col not in input_df.columns:
            if training_data[col].dtype in ['int64', 'float64']:
                input_df[col] = training_data[col].median()
            else:
                input_df[col] = training_data[col].mode()[0]
                
    input_df = input_df[[c for c in training_data.columns if c != 'fraud_reported']]
    X, _ = preprocess_data_max(input_df, fit=False, saved_artifacts=artifacts)
    X = X[artifacts['feature_names']]
    
    prediction = artifacts['model'].predict(X)[0]
    return 'Y' if prediction == 1 else 'N'

if __name__ == "__main__":
    train_max_accuracy()
