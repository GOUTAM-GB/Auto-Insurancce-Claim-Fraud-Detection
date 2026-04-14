import pandas as pd

# Load dataset
df = pd.read_csv(r'media/insurance fraud claims.csv')

# Filter for fraud claims (fraud_reported = 'Y')
fraud_claims = df[df['fraud_reported'] == 'Y'].copy()

print('='*80)
print('FRAUD CLAIMS FROM DATASET (fraud_reported = Y)')
print('='*80)
print(f'Total fraud claims in dataset: {len(fraud_claims)}')
print()

# Select 10 examples
fraud_claims_sample = fraud_claims.sample(n=10, random_state=42)

# Extract required fields for prediction
required_fields = ['policy_number', 'age', 'incident_type', 'collision_type', 'incident_severity', 
                   'authorities_contacted', 'witnesses', 'bodily_injuries', 'total_claim_amount', 
                   'injury_claim', 'property_claim', 'vehicle_claim', 'fraud_reported']

for idx, (i, row) in enumerate(fraud_claims_sample[required_fields].iterrows(), 1):
    print(f'TEST CASE {idx}:')
    print(f'  Policy Number: {int(row["policy_number"])}')
    print(f'  Insured Age: {int(row["age"])}')
    print(f'  Incident Type: {row["incident_type"]}')
    print(f'  Collision Type: {row["collision_type"]}')
    print(f'  Incident Severity: {row["incident_severity"]}')
    print(f'  Authorities Contacted: {row["authorities_contacted"]}')
    print(f'  Witnesses: {int(row["witnesses"])}')
    print(f'  Bodily Injuries: {int(row["bodily_injuries"])}')
    print(f'  Total Claim: ${int(row["total_claim_amount"])}')
    print(f'  Injury Claim: ${int(row["injury_claim"])}')
    print(f'  Property Claim: ${int(row["property_claim"])}')
    print(f'  Vehicle Claim: ${int(row["vehicle_claim"])}')
    print(f'  Expected Result: FRAUD CLAIM DETECTED')
    print()

print('='*80)
