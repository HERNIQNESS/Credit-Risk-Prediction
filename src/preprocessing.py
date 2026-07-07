import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

CATEGORICAL_COLS = ['loan_type', 'New_versus_Repeat']
DROP_COLS = ['ID', 'country_id', 'Total_Amount_to_Repay', 'Lender_portion_to_be_repaid', 'disbursement_date', 'due_date']

def engineer_features(df):
    df = df.copy()
    df['disbursement_date'] = pd.to_datetime(df['disbursement_date'])
    df['due_date'] = pd.to_datetime(df['due_date'])

    df['interest_amount'] = df['Total_Amount_to_Repay'] - df['Total_Amount']
    df['repayment_ratio'] = df['Total_Amount_to_Repay'] / df['Total_Amount']
    df['lender_profit'] = df['Lender_portion_to_be_repaid'] - df['Amount_Funded_By_Lender']
    df['loan_month'] = df['disbursement_date'].dt.month
    df['loan_dayofweek'] = df['disbursement_date'].dt.dayofweek
    df['loan_quarter'] = df['disbursement_date'].dt.quarter

    loan_counts = df.groupby('customer_id')['tbl_loan_id'].count().reset_index()
    loan_counts.columns = ['customer_id', 'customer_loan_count']
    df = df.merge(loan_counts, on='customer_id', how='left')

    df = df.drop(columns=DROP_COLS)
    return df

def build_preprocessor(numerical_cols):
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, CATEGORICAL_COLS)
    ])
    return preprocessor

def get_feature_names(preprocessor, numerical_cols):
    cat_features = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(CATEGORICAL_COLS).tolist()
    return numerical_cols + cat_features

def save_preprocessor(preprocessor, feature_names, path='models/'):
    joblib.dump(preprocessor, path + 'preprocessor.pkl')
    joblib.dump(feature_names, path + 'feature_names.pkl')
    print("Preprocessor saved")

def load_preprocessor(path='models/'):
    preprocessor = joblib.load(path + 'preprocessor.pkl')
    feature_names = joblib.load(path + 'feature_names.pkl')
    return preprocessor, feature_names