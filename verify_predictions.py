import pandas as pd
from utility.data_procesing_max_accuracy import prediction_value

# Load some sample data
data = pd.read_csv(r'media\insurance fraud claims.csv')

# Test with first few rows
for i in range(5):
    row = data.iloc[i].to_dict()
    actual = row['fraud_reported']
    # Remove the target to simulate real input
    # actually prediction_value handles extra columns but let's be clean
    input_data = row.copy()
    del input_data['fraud_reported']
    
    predicted = prediction_value(input_data)
    print(f"Row {i}: Actual={actual}, Predicted={predicted} -> {'PASS' if actual == predicted else 'FAIL'}")
