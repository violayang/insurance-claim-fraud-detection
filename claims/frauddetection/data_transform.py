"""
Data Transformation Module for StateFarm Fraud Detection
Loads data from the database and prepares claim data as a pandas DataFrame
"""

import os
import pandas as pd
from database_connector import DatabaseConnector

def load_claims_as_dataframe(table_name: str = "CLAIMS", limit: int = 1000) -> pd.DataFrame:
    """
    Connects to Oracle DB and loads the specified claims table into a DataFrame.

    Args:
        table_name (str): Name of the claims table.
        limit (int): Maximum rows to load.

    Returns:
        pd.DataFrame: DataFrame with the data.
    """
    # Initialize connector
    db = DatabaseConnector()
    claim_records = db.load_claims_data(table_name=table_name, limit=limit)
    if not claim_records:
        print(f"No records loaded from table {table_name}.")
        return pd.DataFrame()  # return empty dataframe

    df = pd.DataFrame(claim_records)
    print(f"Loaded {len(df)} records from {table_name}.")
    return df

def restructure_claims_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply column renaming and data type conversions on claims DataFrame.

    Args:
        df (pd.DataFrame): Raw claims df.
    Returns:
        pd.DataFrame: Restructured DataFrame.
    """
    # Example: Rename columns for consistency
    rename_map = {
        'CLAIM_ID': 'claim_id',
        'CLAIM_TYPE': 'claim_type',
        'CLAIM_AMOUNT': 'claim_amount',
        'POLICY_HOLDER': 'policy_holder',
        'POLICY_NUMBER': 'policy_number',
        'INCIDENT_DATE': 'incident_date',
        'FILING_DATE': 'filing_date',
        # Add more if needed
    }
    df = df.rename(columns=rename_map)

    # Example: Convert data types
    for col in ['claim_amount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for date_col in ['incident_date', 'filing_date']:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Example: Ensure some columns are str
    for col in ['claim_id', 'policy_holder', 'policy_number']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df

if __name__ == "__main__":
    # Load the claims table as a DataFrame
    claims_df = load_claims_as_dataframe(table_name="CLAIMS", limit=1000)

    if not claims_df.empty:
        print("Raw DataFrame columns:", claims_df.columns.tolist())
        claims_df = restructure_claims_dataframe(claims_df)
        print("Transformed DataFrame columns/types:")
        print(claims_df.dtypes)
        print(claims_df.head())
    else:
        print("No data to transform.")
