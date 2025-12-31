import pandas as pd
import joblib
from preprocess import preprocess_data  # Import the preprocessing function

# Load the trained model
model = joblib.load('../loan_model.pkl')

def predict_loan(data_dict):
    """
    Predict whether a personal loan will be approved.
    
    Parameters:
    data_dict (dict): Input data with keys: Age, Income, CreditScore, LoanAmount, Gender
    
    Returns:
    str: "Approved ✅" or "Denied ❌"
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame([data_dict])
    
    # Preprocess input data
    # Concatenate with dummy target column to use preprocess_data function
    X, _ = preprocess_data(pd.concat([df, pd.DataFrame({"LoanApproved":[0]})], axis=1))
    
    # Predict
    prediction = model.predict(X)
    
    # Return readable output
    if prediction[0] == 1:
        return "Approved ✅"
    else:
        return "Denied ❌"
