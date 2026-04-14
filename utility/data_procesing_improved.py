import os
import io
import base64
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report, roc_auc_score

# XGBoost for better performance on imbalanced data
import xgboost as xgb
from imblearn.over_sampling import SMOTE
try:
    import lightgbm as lgb
except:
    lgb = None

MODEL_PATH = r'models/fraud_detection_model.pkl'
SCALER_PATH = r'models/fraud_scaler.pkl'
ENCODER_PATH = r'models/fraud_encoders.pkl'

def ensure_model_dir():
    """Create models directory if it doesn't exist"""
    os.makedirs(r'models', exist_ok=True)

def load_data(file_path):
    """Load the CSV file"""
    data = pd.read_csv(file_path)
    return data

def preprocess_data_improved(data, fit_encoders=True, encoders=None, scaler=None):
    """
    Enhanced preprocessing with better feature selection and handling
    Uses all available features intelligently
    """
    X = data.drop(columns=['fraud_reported'])
    y = data['fraud_reported']
    
    # Handle missing values
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Impute numeric data with median
    numeric_imputer = SimpleImputer(strategy='median')
    X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
    
    # Impute categorical data with most frequent
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
    
    # Encode categorical variables
    label_encoders = {}
    
    if fit_encoders:
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    else:
        # Use provided encoders
        if encoders:
            for col in categorical_cols:
                if col in encoders:
                    X[col] = encoders[col].transform(X[col].astype(str))
                else:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le
    
    # Feature engineering - create derived features
    X['claim_total_ratio'] = X['total_claim_amount'] / (X['injury_claim'] + X['property_claim'] + X['vehicle_claim'] + 1)
    X['injury_to_property_ratio'] = X['injury_claim'] / (X['property_claim'] + 1)
    X['vehicle_to_total_ratio'] = X['vehicle_claim'] / (X['total_claim_amount'] + 1)
    X['premium_to_claim_ratio'] = X['policy_annual_premium'] / (X['total_claim_amount'] + 1)
    
    # Select more features using mutual information - use all useful features
    k = min(20, X.shape[1])  # Select top 20 features
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Selected Features ({len(selected_features)}): {selected_features}")
    
    # Encode target variable
    if fit_encoders:
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
    else:
        y_encoded = y
    
    # Scale features (important for XGBoost performance)
    if fit_encoders:
        scaler = StandardScaler()
        X_selected = scaler.fit_transform(X_selected)
    else:
        if scaler:
            X_selected = scaler.transform(X_selected)
    
    return X_selected, y_encoded, label_encoders, selector, scaler if fit_encoders else None, selected_features

def train_model_improved(X_train, y_train):
    """
    Train ensemble of models with proper handling of imbalanced data
    Uses XGBoost with optimized parameters
    """
    print(f"Original training set class distribution: {np.bincount(y_train)}")
    
    # Calculate scale_pos_weight for imbalanced data
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    
    print(f"Negative samples: {neg_count}, Positive samples: {pos_count}")
    print(f"Scale position weight: {scale_pos_weight:.2f}")
    
    # Apply SMOTE only on a copy for better balance
    smote = SMOTE(random_state=42, k_neighbors=3)
    try:
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE - Class distribution: {np.bincount(y_train_balanced)}")
    except:
        X_train_balanced, y_train_balanced = X_train, y_train
        print("SMOTE not applied - using original data")
    
    # Train primary XGBoost model with optimized parameters
    model = xgb.XGBClassifier(
        n_estimators=300,           # More trees for better learning
        max_depth=5,                # Shallower trees to prevent overfitting
        learning_rate=0.03,         # Slower learning rate for better generalization
        subsample=0.85,             # Use 85% of training data per iteration
        colsample_bytree=0.9,       # Use 90% of features
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        min_child_weight=1,         # Minimum samples per leaf
        gamma=0.5,                  # Minimum loss reduction for split
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=15    # Stop if no improvement for 15 rounds
    )
    
    # Train on balanced data if available
    model.fit(
        X_train_balanced, y_train_balanced,
        eval_set=[(X_train[:len(X_train)//5], y_train[:len(y_train)//5])],
        verbose=False
    )
    
    print("Model training completed with Optimized XGBoost")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nModel Performance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal (N)', 'Fraud (Y)']))
    
    return accuracy, precision, recall, f1, roc_auc

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image_base64

def create_barplot_image(data):
    """Create bar plot visualization"""
    fig = plt.figure(figsize=(10, 6))
    fraud_counts = data['fraud_reported'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    fraud_counts.plot(kind='bar', color=colors, ax=fig.gca())
    plt.title('Fraud vs Normal Claims Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Fraud Status')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    for i, v in enumerate(fraud_counts):
        fig.gca().text(i, v + 10, str(v), ha='center', fontweight='bold')
    return fig_to_base64(fig)

def create_confusion_matrix_image(conf_matrix, class_labels):
    """Create confusion matrix heatmap"""
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=class_labels, yticklabels=class_labels, cbar=True)
    plt.title('Confusion Matrix - Fraud Detection', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    return fig_to_base64(fig)

def save_model(model, encoders, selector, scaler):
    """Save trained model and preprocessing objects to disk"""
    ensure_model_dir()
    
    model_data = {
        'model': model,
        'encoders': encoders,
        'selector': selector,
        'scaler': scaler
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved successfully to {MODEL_PATH}")

def load_model():
    """Load trained model and preprocessing objects from disk"""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        print(f"Model loaded from {MODEL_PATH}")
        return model_data
    else:
        print(f"Model not found at {MODEL_PATH}")
        return None

# Main execution
def main():
    """Main function for model training and evaluation"""
    file_path = r'media/insurance fraud claims.csv'
    
    # Load and preprocess data
    data = load_data(file_path)
    print(f"Dataset shape: {data.shape}")
    print(f"Fraud distribution:\n{data['fraud_reported'].value_counts()}\n")
    
    X, y, encoders, selector, scaler, selected_features = preprocess_data_improved(data, fit_encoders=True)
    
    print(f'Preprocessing completed. Features selected: {len(selected_features)} out of original features')
    
    # Split data with stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Train-test split completed. Train: {X_train.shape}, Test: {X_test.shape}\n')
    
    # Train model
    model = train_model_improved(X_train, y_train)
    
    # Evaluate model
    accuracy, precision, recall, f1, roc_auc = evaluate_model(model, X_test, y_test)
    
    # Save model and preprocessing objects
    save_model(model, encoders, selector, scaler)
    
    # Create visualizations
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    barplot_image = create_barplot_image(data)
    confusion_image = create_confusion_matrix_image(conf_matrix, ['Normal (N)', 'Fraud (Y)'])
    
    return accuracy, precision, recall, barplot_image, confusion_image

def prediction_value(input_data):
    """
    Predict fraud for a single input sample using saved model
    """
    file_path = r'media/insurance fraud claims.csv'
    
    # Load model and preprocessing objects
    model_data = load_model()
    if model_data is None:
        print("Error: Model not found. Please train the model first.")
        return 'Error'
    
    model = model_data['model']
    encoders = model_data['encoders']
    selector = model_data['selector']
    scaler = model_data['scaler']
    
    # Load full dataset for training data stats
    data = load_data(file_path)
    print('Data loaded successfully')
    
    # Create input DataFrame
    input_df = pd.DataFrame([input_data])
    print("Input DataFrame:")
    print(input_df)
    
    # Prepare training data structure
    training_X = data.drop(columns=['fraud_reported'])
    numeric_cols = training_X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = training_X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create a full feature set by filling missing columns with default values
    for col in training_X.columns:
        if col not in input_df.columns:
            if col in numeric_cols:
                input_df[col] = training_X[col].median()
            else:
                input_df[col] = training_X[col].mode()[0] if len(training_X[col].mode()) > 0 else training_X[col].iloc[0]
    
    # Handle missing values
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    # Process numeric columns
    X_numeric = input_df[numeric_cols].copy()
    X_numeric.iloc[:] = numeric_imputer.fit_transform(X_numeric)
    
    # Process categorical columns
    X_categorical = input_df[categorical_cols].copy()
    
    # Encode categorical values
    for col in categorical_cols:
        if col in encoders:
            try:
                X_categorical[col] = encoders[col].transform(X_categorical[col].astype(str))
            except Exception as e:
                # Handle unknown categories - use most frequent value
                X_categorical[col] = 0
    
    # Combine all features
    X_full = pd.concat([X_numeric, X_categorical], axis=1)
    X_full = X_full[training_X.columns]  # Ensure correct column order
    
    # Add feature engineering (same as training)
    X_full['claim_total_ratio'] = X_full['total_claim_amount'] / (X_full['injury_claim'] + X_full['property_claim'] + X_full['vehicle_claim'] + 1)
    X_full['injury_to_property_ratio'] = X_full['injury_claim'] / (X_full['property_claim'] + 1)
    X_full['vehicle_to_total_ratio'] = X_full['vehicle_claim'] / (X_full['total_claim_amount'] + 1)
    X_full['premium_to_claim_ratio'] = X_full['policy_annual_premium'] / (X_full['total_claim_amount'] + 1)
    
    # Apply feature selection
    X_selected = selector.transform(X_full)
    
    # Scale features
    X_scaled = scaler.transform(X_selected)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    prediction_proba = model.predict_proba(X_scaled)[0]
    
    result = 'Y' if prediction == 1 else 'N'
    confidence = max(prediction_proba) * 100
    
    print(f"\nPredicted value for input: {result}")
    print(f"Confidence: {confidence:.2f}%")
    return result

if __name__ == "__main__":
    accuracy, precision, recall, barplot, confusion = main()
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
