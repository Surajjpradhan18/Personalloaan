import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    # Example preprocessing
    df = df.dropna()
    le = LabelEncoder()
    if 'Gender' in df.columns:
        df['Gender'] = le.fit_transform(df['Gender'])
    scaler = StandardScaler()
    num_cols = ['Age', 'Income', 'CreditScore']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    X = df.drop('LoanApproved', axis=1)
    y = df['LoanApproved']
    return X, y
