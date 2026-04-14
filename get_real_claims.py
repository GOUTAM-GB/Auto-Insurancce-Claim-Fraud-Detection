import pandas as pd

# Load dataset
df = pd.read_csv(r'media/insurance fraud claims.csv')

# Filter for real claims (fraud_reported = 'N')
real_claims = df[df['fraud_reported'] == 'N'].copy()

print('='*80)
print('REAL CLAIMS FROM DATASET (fraud_reported = N)')
print('='*80)
print(f'Total real claims in dataset: {len(real_claims)}')
print()

# Select 10 good examples with variety
real_claims_sample = real_claims.sample(n=10, random_state=42)

# Extract required fields for prediction
required_fields = ['policy_number', 'age', 'incident_type', 'collision_type', 'incident_severity', 
                   'authorities_contacted', 'witnesses', 'bodily_injuries', 'total_claim_amount', 
                   'injury_claim', 'property_claim', 'vehicle_claim', 'fraud_reported']

for idx, (i, row) in enumerate(real_claims_sample[required_fields].iterrows(), 1):
    pn = int(row['policy_number'])
    age = int(row['age'])
    itype = row['incident_type']
    ct = row['collision_type']
    sev = row['incident_severity']
    auth = row['authorities_contacted']
    wit = int(row['witnesses'])
    injuries = int(row['bodily_injuries'])
    total = int(row['total_claim_amount'])
    injury = int(row['injury_claim'])
    prop = int(row['property_claim'])
    veh = int(row['vehicle_claim'])
    fraud = row['fraud_reported']
    
    print(f'Record {idx}:')
    print(f'  policy_number: {pn}')
    print(f'  age: {age}')
    print(f'  incident_type: {itype}')
    print(f'  collision_type: {ct}')
    print(f'  incident_severity: {sev}')
    print(f'  authorities_contacted: {auth}')
    print(f'  witnesses: {wit}')
    print(f'  bodily_injuries: {injuries}')
    print(f'  total_claim_amount: {total}')
    print(f'  injury_claim: {injury}')
    print(f'  property_claim: {prop}')
    print(f'  vehicle_claim: {veh}')
    print(f'  fraud_reported: {fraud}')
    print()

print('='*80)
print('TEST DATA - Use these values in your Prediction Page form:')
print('='*80)
print()

for idx, (i, row) in enumerate(real_claims_sample[required_fields].iterrows(), 1):
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
    print(f'  Expected Result: Real Claim')
    print()
