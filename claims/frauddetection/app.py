"""
StateFarm AI-Powered Insights Dashboard - Backend API
Flask application providing REST API for fraud detection insights
"""

from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from fraud_detection_service import FraudDetectionService, HybridFraudDetectionService
from database_connector import DatabaseConnector, QUERIES
import json
from datetime import datetime, timedelta,date
import random
import subprocess
import csv
from io import StringIO
from typing import List, Dict

app = Flask(__name__)
CORS(app)

# Initialize services
# fraud_service = HybridFraudDetectionService()
fraud_service = FraudDetectionService()
db_connector = DatabaseConnector()

# In-memory storage for demo (use database in production)
analysis_history = []


@app.route('/')
def index():
    """Serve the dashboard homepage"""
    return render_template('dashboard.html')


@app.route('/diagnostics')
def diagnostics():
    """Serve the diagnostics page"""
    return render_template('diagnostics.html')


@app.route('/test')
def test_minimal():
    """Serve the minimal test page"""
    return render_template('test_minimal.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'StateFarm AI Fraud Detection'
    })


@app.route('/api/analyze-claim', methods=['POST'])
def analyze_claim():
    """
    Analyze a single claim for fraud
    
    Expected JSON payload:
    {
        "claim_id": "string",
        "claim_type": "string",
        "claim_amount": number,
        "incident_date": "string",
        "filing_date": "string",
        ... (other claim fields)
    }
    """
    try:
        claim_data = request.json
        
        if not claim_data:
            return jsonify({'error': 'No claim data provided'}), 400
        
        # Validate required fields
        required_fields = ['claim_id', 'claim_type', 'claim_amount']
        missing_fields = [field for field in required_fields if field not in claim_data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Analyze the claim
        analysis = fraud_service.analyze_claim(claim_data)
        
        # Store in history
        analysis_history.append(analysis)
        
        return jsonify(analysis), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Analyze multiple claims in batch
    
    Expected JSON payload:
    {
        "claims": [
            { claim_data_1 },
            { claim_data_2 },
            ...
        ]
    }
    """
    try:
        data = request.json
        claims = data.get('claims', [])
        
        if not claims:
            return jsonify({'error': 'No claims provided'}), 400
        
        results = []
        for claim in claims:
            analysis = fraud_service.analyze_claim(claim)
            results.append(analysis)
            analysis_history.append(analysis)
        
        return jsonify({
            'total_analyzed': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/insights/summary', methods=['GET'])
def get_insights_summary():
    """Get summary insights from all analyzed claims"""
    try:
        if not analysis_history:
            return jsonify({
                'message': 'No claims analyzed yet',
                'total_claims_analyzed': 0,
                'high_risk_count': 0,
                'medium_risk_count': 0,
                'low_risk_count': 0,
                'high_risk_percentage': 0,
                'average_fraud_score': 0,
                'average_confidence': 0,
                'top_fraud_indicators': [],
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # Calculate summary statistics
        summary = fraud_service.get_insights_summary(analysis_history)

        # summary = fraud_service.openai_service.get_insights_summary(analysis_history)   ## uncomment if use HybridFraudDetectionService
        
        return jsonify(summary), 200
        
    except Exception as e:
        print(f"Error in get_insights_summary: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'total_claims_analyzed': 0,
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'low_risk_count': 0
        }), 200  # Return 200 with error data instead of 500


@app.route('/api/insights/trends', methods=['GET'])
def get_trends():
    """Get fraud detection trends over time"""
    try:
        if not analysis_history:
            return jsonify({
                'trends': [],
                'period_start': None,
                'period_end': None
            }), 200
            
        # Group analyses by date
        daily_stats = {}
        
        for analysis in analysis_history:
            timestamp = analysis.get('timestamp', '')
            if timestamp:
                date = timestamp.split('T')[0]
                
                if date not in daily_stats:
                    daily_stats[date] = {
                        'date': date,
                        'total': 0,
                        'high_risk': 0,
                        'medium_risk': 0,
                        'low_risk': 0,
                        'avg_fraud_score': 0,
                        'total_fraud_score': 0
                    }
                
                stats = daily_stats[date]
                stats['total'] += 1
                
                risk_level = analysis.get('risk_level', 'LOW')
                if risk_level in ['HIGH', 'CRITICAL']:
                    stats['high_risk'] += 1
                elif risk_level == 'MEDIUM':
                    stats['medium_risk'] += 1
                else:
                    stats['low_risk'] += 1
                
                stats['total_fraud_score'] += analysis.get('fraud_score', 0)
        
        # Calculate averages
        for date, stats in daily_stats.items():
            if stats['total'] > 0:
                stats['avg_fraud_score'] = round(stats['total_fraud_score'] / stats['total'], 2)
            del stats['total_fraud_score']
        
        # Convert to list and sort by date
        trends = sorted(daily_stats.values(), key=lambda x: x['date'])
        
        return jsonify({
            'trends': trends,
            'period_start': trends[0]['date'] if trends else None,
            'period_end': trends[-1]['date'] if trends else None
        }), 200
        
    except Exception as e:
        print(f"Error in get_trends: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'trends': [],
            'period_start': None,
            'period_end': None
        }), 200  # Return 200 with empty data instead of 500


@app.route('/api/insights/high-risk-claims', methods=['GET'])
def get_high_risk_claims():
    """Get all high-risk claims"""
    try:
        if not analysis_history:
            return jsonify({
                'total_high_risk': 0,
                'claims': []
            }), 200
            
        high_risk = [
            analysis for analysis in analysis_history 
            if analysis.get('risk_level') in ['HIGH', 'CRITICAL']
        ]
        
        # Sort by fraud score (highest first)
        high_risk.sort(key=lambda x: x.get('fraud_score', 0), reverse=True)
        
        return jsonify({
            'total_high_risk': len(high_risk),
            'claims': high_risk
        }), 200
        
    except Exception as e:
        print(f"Error in get_high_risk_claims: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'total_high_risk': 0,
            'claims': []
        }), 200  # Return 200 with empty data instead of 500


@app.route('/api/insights/top-indicators', methods=['GET'])
def get_top_indicators():
    """Get most common fraud indicators"""
    try:
        if not analysis_history:
            return jsonify({
                'top_indicators': [],
                'total_unique_indicators': 0
            }), 200
            
        # Collect all indicators
        indicator_counts = {}
        
        for analysis in analysis_history:
            indicators = analysis.get('fraud_indicators', [])
            for indicator in indicators:
                indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1
        
        # Sort by frequency
        top_indicators = sorted(
            [{'indicator': k, 'count': v} for k, v in indicator_counts.items()],
            key=lambda x: x['count'],
            reverse=True
        )[:15]
        
        return jsonify({
            'top_indicators': top_indicators,
            'total_unique_indicators': len(indicator_counts)
        }), 200
        
    except Exception as e:
        print(f"Error in get_top_indicators: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'top_indicators': [],
            'total_unique_indicators': 0
        }), 200  # Return 200 with empty data instead of 500


@app.route('/api/test-data/generate', methods=['POST'])
def generate_test_data():
    """Generate sample test data for demonstration"""
    try:
        count = request.json.get('count', 10)
        
        # Generate random test claims
        test_claims = []
        claim_types = ['Auto Collision', 'Auto Theft', 'Property Damage', 'Personal Injury']
        locations = ['Los Angeles, CA', 'Chicago, IL', 'New York, NY', 'Houston, TX', 'Phoenix, AZ']
        
        for i in range(count):
            days_ago = random.randint(1, 90)
            incident_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            filing_delay = random.randint(0, 30)
            filing_date = (datetime.now() - timedelta(days=days_ago - filing_delay)).strftime('%Y-%m-%d')
            
            claim = {
                'claim_id': f'CLM-2024-{1000 + i}',
                'claim_type': random.choice(claim_types),
                'claim_amount': random.randint(5000, 50000),
                'incident_date': incident_date,
                'filing_date': filing_date,
                'policy_holder': f'Test Customer {i+1}',
                'policy_number': f'POL-{100000 + i}',
                'policy_start_date': (datetime.now() - timedelta(days=random.randint(30, 730))).strftime('%Y-%m-%d'),
                'previous_claims_count': random.randint(0, 5),
                'years_as_customer': round(random.uniform(0.1, 10.0), 1),
                'incident_location': random.choice(locations),
                'incident_description': f'Test incident description for claim {i+1}',
                'witnesses': 'Yes' if random.random() > 0.5 else 'No',
                'police_report_filed': 'Yes' if random.random() > 0.6 else 'No',
                'days_since_policy_start': random.randint(1, 365),
                'filing_delay_days': filing_delay,
                'similar_claims_in_area': random.randint(0, 10),
                'repair_provider': f'Provider {random.randint(1, 5)}'
            }
            test_claims.append(claim)
        
        return jsonify({
            'message': f'Generated {count} test claims',
            'claims': test_claims
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_analysis_history():
    """Get all analysis history"""
    try:
        limit = request.args.get('limit', type=int, default=100)
        
        return jsonify({
            'total': len(analysis_history),
            'history': analysis_history[-limit:] if limit > 0 else analysis_history
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear analysis history"""
    try:
        global analysis_history
        count = len(analysis_history)
        analysis_history = []
        
        return jsonify({
            'message': f'Cleared {count} records',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/database/connect', methods=['POST'])
def database_connect():
    """Connect to Oracle database"""
    try:
        success = db_connector.connect()
        if success:
            # Get available tables
            tables = db_connector.get_claims_tables()
            return jsonify({
                'status': 'connected',
                'connection': db_connector.connection_string,
                'available_tables': tables,
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({'error': 'Failed to connect to database'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/database/tables', methods=['GET'])
def get_database_tables():
    """Get list of available claims tables"""
    try:
        if not db_connector.is_connected:
            db_connector.connect()
        
        tables = db_connector.get_claims_tables()
        
        # Get row counts for each table
        table_info = []
        for table_name in tables:
            count = db_connector.count_records(table_name)
            table_info.append({
                'name': table_name,
                'row_count': count
            })
        
        return jsonify({
            'tables': table_info,
            'connection': db_connector.connection_string
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500






# ---------------- BATCH FROM DATABASE BUTTON - BEGIN---------------------
@app.route('/api/database/batch-from-database', methods=['POST'])
def batch_from_database():
    """
    Trigger batch operation from database.
    This is a placeholder endpoint for the 'Batch from Database' button.
    """
    # You can add real logic here later
    print("Batch from Database endpoint triggered.")
    return jsonify({"message": "Batch from Database endpoint is not yet implemented."}), 200

# ---------------- BATCH FROM DATABASE BUTTON - END---------------------



@app.route('/api/database/load-claims', methods=['POST'])
def load_claims_from_database():
    """
    Load claims from database and analyze them
    
    Expected JSON:
    {
        "table_name": "CLAIMS",
        "limit": 50,
        "auto_analyze": true
    }
    """
    try:
        data = request.json
        table_name = data.get('table_name', 'CLAIMS')
        limit = data.get('limit', 20)
        auto_analyze = data.get('auto_analyze', True)
        
        # Ensure connection
        if not db_connector.is_connected:
            if not db_connector.connect():
                return jsonify({'error': 'Failed to connect to database'}), 500
        
        # Load only non-approved claims from database (exclude 'Approved' and 'Closed')
        print(f"Loading {limit} non-approved claims from V_ACTIVE_CLAIMS...")
        db_records = db_connector.load_claims_data(table_name, limit, include_approved=False)
        print(f"Loaded claims from {table_name}")

        if not db_records:
            # Fallback: Generate sample data for demonstration
            print("No non-approved claims in database, generating sample data...")
            # db_records = generate_sample_claims_from_db(limit)
        
        print(f"Retrieved {len(db_records)} non-approved records from database")
        
        # Transform to fraud detection format
        transformed_claims = []
        for record in db_records:
            try:
                claim = db_connector.transform_db_record_to_claim(record)
                # dev - check claim object
                # print("claim type: ", type(claim))
                # print("claim: \n", claim)

                transformed_claims.append(claim)
            except Exception as e:
                print(f"Error transforming record: {e}")
                continue
        
        print(f"Transformed {len(transformed_claims)} claims")
        
        # Optionally analyze claims
        analyses = []
        if auto_analyze and transformed_claims:
            # Limit to reasonable number for analysis
            claims_to_analyze = transformed_claims[:min(20, len(transformed_claims))]
            print(f"Analyzing {len(claims_to_analyze)} claims...")
            
            for i, claim in enumerate(claims_to_analyze):
                try:
                    print(f"  Analyzing claim {i+1}/{len(claims_to_analyze)}: {claim.get('claim_id')}")
                    analysis = fraud_service.analyze_claim(claim)
                    analyses.append(analysis)
                    analysis_history.append(analysis)
                except Exception as e:
                    print(f"Error analyzing claim {claim.get('claim_id')}: {e}")
        
        print(f"Analysis complete: {len(analyses)} claims analyzed")
        
        return jsonify({
            'message': f'Loaded {len(transformed_claims)} claims from {table_name}',
            'claims_loaded': len(transformed_claims),
            'claims_analyzed': len(analyses),
            'table_name': table_name,
            'analyses': analyses[:10],  # Return first 10 analyses
            'high_risk_count': sum(1 for a in analyses if a.get('risk_level') in ['HIGH', 'CRITICAL']),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"Error in load_claims_from_database: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/database/load-claims-progressive', methods=['POST'])
def load_claims_progressive():
    """
    Load and analyze claims with real-time progress updates using Server-Sent Events
    """
    try:
        # Get request data before creating generator
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        table_name = data.get('table_name', 'CLAIMS')
        limit = data.get('limit', 2)
        auto_analyze = data.get('auto_analyze', True)
        
        def generate():
            try:
                # Send connection status
                yield f"data: {json.dumps({'type': 'connecting', 'message': 'Connecting to database...'})}\n\n"
                
                # Ensure connection
                if not db_connector.is_connected:
                    if not db_connector.connect():
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to connect to database'})}\n\n"
                        return
                
                yield f"data: {json.dumps({'type': 'connected', 'message': 'Connected successfully'})}\n\n"
                
                # Load only non-approved claims (exclude 'Approved' and 'Closed')
                yield f"data: {json.dumps({'type': 'loading', 'message': f'Loading non-approved claims from {table_name}...'})}\n\n"
                db_records = db_connector.load_claims_data(table_name, limit, include_approved=False)

                ## TODO:  add code append 1)incident date, 2)location, 3)anyone injured,  4)police report filed

                # if not db_records:
                #     db_records = generate_sample_claims_from_db(limit)
                
                yield f"data: {json.dumps({'type': 'loaded', 'count': len(db_records), 'message': f'Loaded {len(db_records)} non-approved claims'})}\n\n"
                
                # Transform claims
                transformed_claims = []
                for i, record in enumerate(db_records):
                    try:
                        print("record is ???\n", record)
                        claim = db_connector.transform_db_record_to_claim(record)
                        print("claim is ???\n", claim)
                        transformed_claims.append(claim)
                        
                        yield f"data: {json.dumps({'type': 'claim_loaded', 'claim_id': claim.get('claim_id'), 'claim_type': claim.get('claim_type'), 'amount': claim.get('claim_amount'), 'current': i+1, 'total': len(db_records)})}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Error transforming claim: {str(e)}'})}\n\n"
                
                # Analyze claims
                if auto_analyze and transformed_claims:
                    analyses = []
                    high_risk = 0
                    medium_risk = 0
                    low_risk = 0
                    
                    claims_to_analyze = transformed_claims[:min(limit, len(transformed_claims))]
                    
                    for i, claim in enumerate(claims_to_analyze):
                        try:
                            yield f"data: {json.dumps({'type': 'claim_analyzing', 'claim_id': claim.get('claim_id'), 'current': i+1, 'total': len(claims_to_analyze)})}\n\n"
                            
                            analysis = fraud_service.analyze_claim(claim)
                            analyses.append(analysis)
                            analysis_history.append(analysis)
                            
                            # Count risk levels
                            risk_level = analysis.get('risk_level', 'LOW')
                            if risk_level in ['HIGH', 'CRITICAL']:
                                high_risk += 1
                            elif risk_level == 'MEDIUM':
                                medium_risk += 1
                            else:
                                low_risk += 1
                            
                            yield f"data: {json.dumps({'type': 'claim_analyzed', 'claim_id': claim.get('claim_id'), 'risk_level': risk_level, 'fraud_score': analysis.get('fraud_score'), 'current': i+1, 'total': len(claims_to_analyze), 'analysis': analysis})}\n\n"
                            
                        except Exception as e:
                            print(f"Error analyzing claim: {e}")
                            yield f"data: {json.dumps({'type': 'error', 'message': f'Error analyzing claim: {str(e)}'})}\n\n"
                    
                    # Send completion
                    yield f"data: {json.dumps({'type': 'complete', 'total_analyzed': len(analyses), 'high_risk_count': high_risk, 'medium_risk_count': medium_risk, 'low_risk_count': low_risk})}\n\n"
                
            except Exception as e:
                print(f"Error in generator: {e}")
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        response = Response(generate(), mimetype='text/event-stream')
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['X-Accel-Buffering'] = 'no'
        return response
        
    except Exception as e:
        print(f"Error in load_claims_progressive: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def parse_sqlcl_csv_output(csv_output: str, table_name: str) -> List[Dict]:
    """Parse CSV output from SQLcl"""
    try:
        # Find CSV data in output
        lines = csv_output.split('\n')
        csv_data = []
        in_csv = False
        
        for line in lines:
            if ',' in line and not line.startswith('SQL>'):
                in_csv = True
            if in_csv:
                csv_data.append(line)
        
        if not csv_data:
            return []
        
        # Parse CSV
        csv_text = '\n'.join(csv_data)
        csv_reader = csv.DictReader(StringIO(csv_text))
        return list(csv_reader)
        
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return []


def generate_sample_claims_from_db(count: int = 10) -> List[Dict]:
    """Generate sample claims data (fallback when DB not available)"""
    claims = []
    claim_types = ['Auto Collision', 'Auto Theft', 'Property Damage', 'Personal Injury', 'Fire Damage']
    
    for i in range(count):
        days_ago = random.randint(1, 365)
        incident_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        filing_delay = random.randint(0, 30)
        filing_date = (datetime.now() - timedelta(days=days_ago - filing_delay)).strftime('%Y-%m-%d')
        policy_start_date = datetime.now() - timedelta(days=random.randint(30, 1095))
        
        claim = {
            'CLAIM_ID': f'DB-CLM-{1000 + i}',
            'CLAIM_TYPE': random.choice(claim_types),
            'CLAIM_AMOUNT': random.randint(5000, 75000),
            'INCIDENT_DATE': incident_date,
            'FILING_DATE': filing_date,
            'POLICY_HOLDER': f'DB Customer {i+1}',
            'POLICY_NUMBER': f'DB-POL-{200000 + i}',
            'POLICY_START_DATE': policy_start_date.strftime('%Y-%m-%d'),
            'PREVIOUS_CLAIMS': random.randint(0, 4),
            # 'YEARS_AS_CUSTOMER': round(random.uniform(0.1, 10.0), 1),
            'YEARS_AS_CUSTOMER': round((datetime.now().date() - policy_start_date.date()).days / 365.25, 2),
            'INCIDENT_LOCATION': random.choice(['Los Angeles, CA', 'Chicago, IL', 'New York, NY', 'Houston, TX']),
            'INCIDENT_DESCRIPTION': f'Database claim incident {i+1}',
            'WITNESSES': 'Yes' if random.random() > 0.5 else 'No',
            'POLICE_REPORT': 'Yes' if random.random() > 0.4 else 'No',
            'DAYS_SINCE_POLICY_START': random.randint(30, 1095),
            'FILING_DELAY': filing_delay,
            'SIMILAR_CLAIMS': random.randint(0, 8),
            'REPAIR_PROVIDER': f'Provider {random.randint(1, 10)}'
        }
        claims.append(claim)
    
    return claims


if __name__ == '__main__':
    print("Starting StateFarm AI-Powered Insights Dashboard...")
    print("Dashboard will be available at: http://localhost:3200")
    app.run(debug=True, host='0.0.0.0', port=3200)