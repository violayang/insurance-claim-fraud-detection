import requests
import oci
import os
from oci.signer import Signer
import json
import oracledb
from dotenv import load_dotenv
import logging
from sqlalchemy import create_engine
import pandas as pd

logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Read database credentials from environment
username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
dsn = os.getenv("DB_CONNECTION_STRING")
wallet_location = os.getenv("WALLET_LOCATION")
tns_admin = os.getenv("TNS_ADMIN")
wallet_password = os.getenv("WALLET_PASSWORD")

print("Wallet file: ", wallet_location+"/Wallet_vibecoding.zip")
print(os.getcwd())

# Test DB connection --- optional
# connection = oracledb.connect(
#     user=username,
#     password=password,
#     dsn=dsn,
#     config_dir=tns_admin,
#     wallet_location=tns_admin,
#     wallet_password=wallet_password
# )
# print("Successfully connected to the Oracle database use {} db user".format(username))


# Create Connection
def get_oracle_conn():
    return oracledb.connect(
        user=username,
        password=password,
        dsn=dsn,
        config_dir=tns_admin,
        wallet_location=tns_admin,
        wallet_password=wallet_password
    )

# Create SQLAlchemy engine using the creator
engine = create_engine('oracle+oracledb://', creator=get_oracle_conn)

# Read table to Dataframe
sql_statement = """SELECT * FROM insurance_data"""
df = pd.read_sql(sql_statement, con=engine)

print("Successfully loaded data from Oracle database")


# Data Transformation




'''
config = oci.config.from_file("~/.oci/config") # replace with the location of your oci config file
auth = Signer(
  tenancy=config['tenancy'],
  user=config['user'],
  fingerprint=config['fingerprint'],
  private_key_file_location=config['key_file']
)

endpoint = 'https://modeldeployment.us-chicago-1.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.us-chicago-1.amaaaaaawe6j4fqapcylmoy5g2wvyiinnr2twaca4jm33r6suoagyjehbogq/predict'
# body = {} # payload goes here
body = X_test.head(0).values.tolist()
headers = {} # header goes here
requests.post(endpoint, json=body, auth=auth, headers=headers).json()

'''

# # close the connection
# connection.close()