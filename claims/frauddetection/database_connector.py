"""
Database Integration Module for StateFarm Fraud Detection
Connects to Oracle VIBECODING_MEDIUM database and loads claims data
"""

import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

try:
    import oracledb
    ORACLE_CLIENT_AVAILABLE = True
except ImportError:
    try:
        import cx_Oracle as oracledb
        ORACLE_CLIENT_AVAILABLE = True
    except ImportError:
        ORACLE_CLIENT_AVAILABLE = False
        print("Warning: Neither oracledb nor cx_Oracle is installed. Install with: pip install oracledb")


class DatabaseConnector:
    """
    Handles connection to Oracle database and claims data retrieval
    Uses python-oracledb or cx_Oracle for database operations
    """
    
    def __init__(self):
        # Get credentials from environment
        self.tns_admin = os.getenv('TNS_ADMIN')
        self.connection_string = os.getenv('DB_CONNECTION_STRING', 'VIBECODING_MEDIUM')
        self.username = os.getenv('DB_USERNAME', 'ADMIN')
        self.password = os.getenv('DB_PASSWORD')
        self.wallet_password = os.getenv('WALLET_PASSWORD')
        
        self.connection = None
        self.is_connected = False
        
        # Initialize Oracle client
        if ORACLE_CLIENT_AVAILABLE and self.tns_admin:
            try:
                # Set TNS_ADMIN environment variable
                os.environ['TNS_ADMIN'] = self.tns_admin
                print(f"TNS_ADMIN set to: {self.tns_admin}")
            except Exception as e:
                print(f"Error setting TNS_ADMIN: {e}")
    
    def connect(self) -> bool:
        """Connect to the Oracle database using wallet credentials"""
        if not ORACLE_CLIENT_AVAILABLE:
            print("Oracle client library not available")
            return False
        
        if not self.password:
            print("Error: DB_PASSWORD not set in .env file")
            return False
            
        try:
            # For Autonomous Database with wallet
            # Need username, password, and connection string with wallet location
            
            self.connection = oracledb.connect(
                user=self.username,
                password=self.password,
                dsn=self.connection_string,
                config_dir=self.tns_admin,
                wallet_location=self.tns_admin,
                wallet_password=self.wallet_password
            )
            
            self.is_connected = True
            print(f"✓ Connected to Oracle database: {self.connection_string}")
            return True
            
        except Exception as e:
            print(f"✗ Database connection failed: {str(e)}")
            self.is_connected = False
            return False
    
    def execute_query(self, query: str) -> List[Dict]:
        """
        Execute a SQL query and return results as list of dictionaries
        
        Args:
            query: SQL query to execute
            
        Returns:
            List of row dictionaries
        """
        if not self.is_connected or not self.connection:
            if not self.connect():
                return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            # Get column names
            columns = [col[0] for col in cursor.description]
            
            # Fetch all rows
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    row_dict[col] = row[i]
                results.append(row_dict)
            
            cursor.close()
            return results
            
        except Exception as e:
            print(f"Query execution error: {str(e)}")
            return []
    
    def get_claims_tables(self) -> List[str]:
        """Get list of claims-related tables"""
        if not self.is_connected:
            if not self.connect():
                return []
        
        try:
            query = """
                SELECT table_name 
                FROM user_tables 
                WHERE LOWER(table_name) LIKE '%claim%'
                ORDER BY table_name
            """
            results = self.execute_query(query)
            return [row['TABLE_NAME'] for row in results]
        except Exception as e:
            print(f"Error getting tables: {e}")
            return ['CLAIMS', 'INSURANCE_CLAIMS', 'CLAIM_DETAILS']
    
    def load_claims_data(self, table_name: str = None, limit: int = 2, include_approved: bool = False) -> List[Dict]:
        """
        Load claims data from database
        
        Args:
            table_name: Specific table to query (optional)
            limit: Maximum number of records to fetch
            include_approved: If False, exclude approved claims (default: False)
            
        Returns:
            List of claim dictionaries
        """
        if not table_name:
            # Get first claims table
            tables = self.get_claims_tables()
            table_name = tables[0] if tables else 'CLAIMS'
        
        try:
            # Only load claims that are NOT approved (pending investigation/review)
            if include_approved:
                # Load all claims
                query = f"""
                    SELECT * FROM {table_name}
                    WHERE LOWER(STATUS) = 'pending'
                    AND ROWNUM <= 5
                    ORDER BY REPORTED_DATE DESC
                    
                """
            else:
                # Only load non-approved claims (exclude 'Approved' and 'Closed' status)
                query = f"""
                    SELECT * FROM {table_name}
                    WHERE STATUS NOT IN ('Approved', 'Closed')
                    AND ROWNUM <= 5
                    ORDER BY INCIDENT_DATE DESC
                """
            
            print(f"Loading claims with filter: include_approved={include_approved}")
            return self.execute_query(query)
        except Exception as e:
            print(f"Error loading claims: {e}")
            return []
    
    def get_table_schema(self, table_name: str) -> List[Dict]:
        """Get schema information for a table"""
        try:
            query = f"""
                SELECT column_name, data_type, data_length, nullable
                FROM user_tab_columns
                WHERE table_name = '{table_name.upper()}'
                ORDER BY column_id
            """
            return self.execute_query(query)
        except Exception as e:
            print(f"Error getting schema: {e}")
            return []
    
    def count_records(self, table_name: str) -> int:
        """Count total records in a table"""
        try:
            query = f"SELECT COUNT(*) as total FROM {table_name}"
            result = self.execute_query(query)
            return result[0]['TOTAL'] if result else 0
        except Exception as e:
            print(f"Error counting records: {e}")
            return 0
    
    def transform_db_record_to_claim(self, record: Dict) -> Dict:
        """
        Transform database record to claim format expected by fraud detection service
        
        Args:
            record: Raw database record
            
        Returns:
            Formatted claim dictionary
        """
        # Map database columns to claim format
        # Handles various column name variations
        
        claim = {
            'claim_id': str(record.get('CLAIM_ID', record.get('ID', record.get('CLAIMID', 'DB-UNKNOWN')))),
            'claim_type': str(record.get('CLAIM_TYPE', record.get('TYPE', record.get('CLAIMTYPE', 'Unknown')))),
            'claim_amount': float(record.get('CLAIM_AMOUNT', record.get('AMOUNT', record.get('TOTAL_AMOUNT', 0)))),
            'incident_date': self._format_date(record.get('INCIDENT_DATE', record.get('DATE_OF_INCIDENT', record.get('INCIDENT_DT')))),
            'filing_date': self._format_date(record.get('FILING_DATE', record.get('DATE_FILED', record.get('FILE_DATE')))),
            'policy_holder': str(record.get('POLICY_HOLDER', record.get('POLICYHOLDER_NAME', record.get('HOLDER', 'Unknown')))),
            'policy_number': str(record.get('POLICY_NUMBER', record.get('POLICY_NO', record.get('POLICY_NUM', 'Unknown')))),
            'policy_start_date': self._format_date(record.get('POLICY_START_DATE', record.get('POLICY_START', record.get('POLICY_START_DT')))),
            'previous_claims_count': int(record.get('PREVIOUS_CLAIMS', record.get('PRIOR_CLAIMS', record.get('PREV_CLAIMS', 0)))),
            'years_as_customer': float(record.get('YEARS_AS_CUSTOMER', record.get('CUSTOMER_YEARS', record.get('TENURE', 0)))),
            'incident_location': str(record.get('INCIDENT_LOCATION', record.get('LOCATION', record.get('ADDRESS', 'Unknown')))),
            'incident_description': str(record.get('INCIDENT_DESCRIPTION', record.get('DESCRIPTION', record.get('DETAILS', 'No description')))),
            'witnesses': str(record.get('WITNESSES', record.get('WITNESS_COUNT', record.get('HAS_WITNESSES', 'Unknown')))),
            'police_report_filed': str(record.get('POLICE_REPORT', record.get('POLICE_REPORT_FILED', record.get('REPORT_FILED', 'Unknown')))),
            'days_since_policy_start': int(record.get('DAYS_SINCE_POLICY_START', record.get('POLICY_AGE_DAYS', 0))),
            'filing_delay_days': int(record.get('FILING_DELAY', record.get('DAYS_TO_FILE', record.get('DELAY_DAYS', 0)))),
            'similar_claims_in_area': int(record.get('SIMILAR_CLAIMS', record.get('AREA_CLAIMS', 0))),
            'repair_provider': str(record.get('REPAIR_PROVIDER', record.get('PROVIDER', record.get('REPAIR_SHOP', 'Unknown'))))
        }
        
        return claim
    
    def _format_date(self, date_value) -> str:
        """Format date to standard string format"""
        if not date_value:
            return datetime.now().strftime('%Y-%m-%d')
        
        if isinstance(date_value, str):
            # Already a string, try to parse and reformat
            try:
                # Try parsing common formats
                from dateutil import parser
                parsed = parser.parse(date_value)
                return parsed.strftime('%Y-%m-%d')
            except:
                return date_value
        
        try:
            # datetime object
            return date_value.strftime('%Y-%m-%d')
        except:
            return datetime.now().strftime('%Y-%m-%d')
    
    def disconnect(self):
        """Disconnect from database"""
        if self.connection:
            try:
                self.connection.close()
                self.is_connected = False
                print("✓ Disconnected from database")
            except Exception as e:
                print(f"Error disconnecting: {e}")
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.disconnect()


# SQL Query Templates
QUERIES = {
    'list_tables': """
        /* LLM in use is claude-sonnet-4 */
        SELECT table_name 
        FROM user_tables 
        WHERE LOWER(table_name) LIKE '%claim%'
        ORDER BY table_name
    """,
    
    'get_claims': """
        /* LLM in use is claude-sonnet-4 */
        SELECT * 
        FROM {table_name}
        WHERE ROWNUM <= {limit}
    """,
    
    'describe_table': """
        /* LLM in use is claude-sonnet-4 */
        SELECT column_name, data_type, data_length, nullable
        FROM user_tab_columns
        WHERE table_name = :table_name
        ORDER BY column_id
    """,
    
    'count_claims': """
        /* LLM in use is claude-sonnet-4 */
        SELECT COUNT(*) as total_claims
        FROM {table_name}
    """
}


def get_available_claim_tables(sql_executor) -> List[str]:
    """
    Get list of available claims tables in the database
    
    Args:
        sql_executor: Function to execute SQL queries
        
    Returns:
        List of table names
    """
    try:
        result = sql_executor(QUERIES['list_tables'])
        print("SQL query statement to load claims from table: \nQUERIES['list_tables']", )
        return [row['TABLE_NAME'] for row in result]
    except Exception as e:
        print(f"Error getting tables: {str(e)}")
        return []


def load_claims_from_table(sql_executor, table_name: str, limit: int = 2) -> List[Dict]:
    """
    Load claims from specified table
    
    Args:
        sql_executor: Function to execute SQL queries
        table_name: Name of the table to query
        limit: Maximum records to fetch
        
    Returns:
        List of claim records
    """
    try:

        query = QUERIES['get_claims'].format(table_name=table_name, limit=limit)
        result = sql_executor(query)
        return result
    except Exception as e:
        print(f"Error loading claims: {str(e)}")
        return []


if __name__ == "__main__":
    # Test database connector
    db = DatabaseConnector()
    print(f"Connection: {db.connection_name}")
    print(f"Available tables: {db.get_claims_tables()}")