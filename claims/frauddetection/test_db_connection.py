#!/usr/bin/env python3
"""
Test Oracle Database Connection
Verifies connection to VIBECODING_MEDIUM database
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("Oracle Database Connection Test")
print("=" * 60)
print()

# Check environment variables
print("1. Checking environment variables...")
tns_admin = os.getenv('TNS_ADMIN')
db_connection = os.getenv('DB_CONNECTION_STRING')
db_username = os.getenv('DB_USERNAME')
db_password = os.getenv('DB_PASSWORD')
wallet_password = os.getenv('WALLET_PASSWORD')
wallet_location = os.getenv('WALLET_LOCATION')

print("ADB wallet location: ",os.environ['WALLET_LOCATION'])
print("files in: ", os.listdir(os.environ['TNS_ADMIN']))

if not tns_admin:
    print("   ✗ TNS_ADMIN not set in .env file")
    sys.exit(1)
else:
    print(f"   ✓ TNS_ADMIN: {tns_admin}")

if not db_connection:
    print("   ✗ DB_CONNECTION_STRING not set in .env file")
    sys.exit(1)
else:
    print(f"   ✓ DB_CONNECTION_STRING: {db_connection}")

if not db_username:
    print("   ✗ DB_USERNAME not set in .env file")
    sys.exit(1)
else:
    print(f"   ✓ DB_USERNAME: {db_username}")

if not db_password:
    print("   ✗ DB_PASSWORD not set in .env file")
    sys.exit(1)
else:
    print(f"   ✓ DB_PASSWORD: {'*' * len(db_password)}")

if not wallet_password:
    print("   ✗ WALLET_PASSWORD not set in .env file")
    sys.exit(1)
else:
    print(f"   ✓ WALLET_PASSWORD: {'*' * len(wallet_password)}")

print()

# Check wallet files
print("2. Checking wallet files...")
if not os.path.exists(tns_admin):
    print(f"   ✗ Wallet directory not found: {tns_admin}")
    sys.exit(1)
else:
    print(f"   ✓ Wallet directory exists: {tns_admin}")

required_files = ['cwallet.sso', 'tnsnames.ora', 'sqlnet.ora']
for filename in required_files:
    filepath = os.path.join(tns_admin, filename)
    if os.path.exists(filepath):
        print(f"   ✓ {filename} exists")
    else:
        print(f"   ✗ {filename} missing")

print()

# Check oracledb package
print("3. Checking oracledb package...")
try:
    import oracledb
    print(f"   ✓ oracledb installed (version {oracledb.__version__})")
except ImportError:
    print("   ✗ oracledb not installed")
    print("   Install with: pip install oracledb==2.0.1")
    sys.exit(1)

print()

# Test database connection
print("4. Testing database connection...")
try:
    from database_connector import DatabaseConnector
    
    db = DatabaseConnector()
    
    # Check if this is the updated version
    if not hasattr(db, 'connection_string'):
        print("   ✗ ERROR: You're using an old version of database_connector.py")
        print()
        print("   Please download and replace database_connector.py:")
        print("   1. Download from: computer:///mnt/user-data/outputs/database_connector.py")
        print("   2. Replace your current database_connector.py")
        print("   3. Run this test again")
        print()
        sys.exit(1)
    
    print(f"   Connecting to: {db.connection_string}")
    
    if db.connect():
        print("   ✓ Successfully connected to database!")
        
        # Try to get tables
        print()
        print("5. Querying available tables...")
        try:
            tables = db.get_claims_tables()
            if tables:
                print(f"   ✓ Found {len(tables)} claims-related tables:")
                for table in tables:
                    print(f"      - {table}")
            else:
                print("   ⚠ No claims tables found (or no tables with 'claim' in name)")
        except Exception as e:
            print(f"   ⚠ Could not query tables: {e}")
        
        # Try to count records
        if tables:
            print()
            print("6. Checking table row counts...")
            for table in tables[:3]:  # Check first 3 tables
                try:
                    count = db.count_records(table)
                    print(f"   ✓ {table}: {count} rows")
                except Exception as e:
                    print(f"   ⚠ {table}: Could not count rows - {e}")
        
        # Disconnect
        db.disconnect()
        
        print()
        print("=" * 60)
        print("✅ SUCCESS! Database connection is working.")
        print("=" * 60)
        print()
        print("You can now:")
        print("1. Start the application: python app.py")
        print("2. Open dashboard: http://localhost:3200")
        print("3. Click 'Load from Database' button")
        print()
        
    else:
        print("   ✗ Failed to connect to database")
        print()
        print("Troubleshooting:")
        print("1. Check wallet files are correct")
        print("2. Verify DB_CONNECTION_STRING matches tnsnames.ora")
        print("3. Confirm WALLET_PASSWORD is correct")
        print("4. Review ORACLE_SETUP.md for more help")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    print()
    print("Troubleshooting:")
    print("1. Ensure database_connector.py exists in current directory")
    print("2. Check .env file has correct credentials")
    print("3. Review ORACLE_SETUP.md for detailed guide")
    import traceback
    traceback.print_exc()
    sys.exit(1)