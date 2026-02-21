"""
StateFarm AI-Powered Fraud Detection Service
Uses OCI GenAI for intelligent fraud analysis replacing heuristic logic
"""
import logging
logging.basicConfig(level=logging.INFO)

import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import oci
from oci.signer import Signer
import requests
from oci.generative_ai_inference.models import ChatDetails
import oci_openai
from oci_openai import OciOpenAI, OciUserPrincipalAuth


# Load environment variables
load_dotenv()

# OCI Config
config = oci.config.from_file("~/.oci/config") # replace with the location of your oci config file
# config = oci.config.from_file((os.getcwd()+"/.oci/config"))   # oci config not in default path
auth = Signer(
  tenancy=config['tenancy'],
  user=config['user'],
  fingerprint=config['fingerprint'],
  private_key_file_location=config['key_file']
)

class FraudDetectionService:
    """
    AI-powered fraud detection service using OCI Generative AI
    Replaces heuristic-based logic with advanced language model analysis
    """


    def __init__(self):
        # OCI GenAI setup
        oci_config_path = os.getenv('OCI_CONFIG_FILE')
        oci_profile = os.getenv('OCI_PROFILE', 'DEFAULT')
        self.oci_enabled = False
        try:
            self.oci_config = oci.config.from_file(oci_config_path, oci_profile)
            self.oci_genai_endpoint = os.getenv('OCI_GENAI_ENDPOINT')  # e.g. "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
            self.oci_compartment_id = os.getenv('OCI_COMPARTMENT_ID')
            self.fd_model_endpoint = os.getenv('MODEL_DEPLOYMENT_ENDPOINT')
            self.oci_model_id = os.getenv('OCI_GENAI_MODEL_ID')  # Model OCID
            # print("oci genai service endpoint, compartment id, model id all loaded")
            if self.oci_genai_endpoint and self.oci_compartment_id and self.oci_model_id:
                self.oci_genai_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                    config=self.oci_config,
                    service_endpoint=self.oci_genai_endpoint
                )
                self.oci_enabled = True

            self.up_auth = OciUserPrincipalAuth(profile_name="chicago")
            self.sync_client = OciOpenAI(
                base_url= self.oci_genai_endpoint+"/20231130/actions/v1",  #"https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/v1"
                auth=self.up_auth,
                compartment_id=self.oci_compartment_id
            )

        except Exception as e:
            print(f"Warning: OCI Generative AI not fully configured: {e}")
            self.oci_genai_client = None


    def test_agent_hub(self):
        """
        test Agent Hub sdk
        """
        try:
            completion = self.sync_client.chat.completions.create(
                model="openai.gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": "How do I output all files in a directory using Python?"
                    }
                ]
            )
            print(completion.choices[0].message.content)
            return completion.model_dump_json()

        except Exception as e:
            print(f"Error: OCI Generative AI Agent Hub if not ready: {e}")


    def claim_fraud_detect(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify if claim is potential fraud use classic machine learning model.
        ML model:
            Isolation Forest - unsupervised anomaly detection
            Model Endpoint - OCI DS Model Deployment
        """

        try:
            body = claim_df.values.tolist()
            headers = {}  # header goes here
            detect_result = requests.post(self.fd_model_endpoint, json=body, auth=auth, headers=headers).json()
            return detect_result['prediction']

        except Exception as e:
            print(f"Error transform claim data: {e}")


        
    def analyze_claim(self, claim_data: Dict) -> Dict:
        """
        Analyze an insurance claim for fraud using OCI GenAI.
        Args:
            claim_data: Dictionary containing claim information
        Returns:
            Dictionary with fraud analysis results
        """

        if not self.oci_enabled:
            return {
                'error': 'OCI Generative AI is not configured or enabled.',
                'fraud_score': 0,
                'risk_level': 'ERROR',
                'timestamp': datetime.now().isoformat()
            }
        try:
            prompt = self._create_fraud_analysis_prompt(claim_data)
            # print(" The claim prompt :\n", prompt)

            # Compose the chat detail for OCI GenAI gpt-oss model
            system_prompt = """You are an expert insurance fraud detection analyst for InsuranceCo. 
                                    Analyze claims for potential fraud indicators and provide detailed insights.
                                    There's an additional Machine Learning model trained based on historical insurance claims to detect if a new submitted claim is outliers or not.
                                    Take the ML model result in consideration when analyzing claims.
                                    Return your analysis in JSON format with the following structure:
                                    {
                                        "fraud_score": <float 0-100>,
                                        "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>",
                                        "fraud_indicators": [<list of detected indicators>],
                                        "confidence": <float 0-100>,
                                        "reasoning": "[<list of explanations>]",
                                        "recommended_actions": [<list of recommended actions>],
                                        "red_flags": [<specific concerning patterns>],
                                        "anomaly_detected": <bool result from ML model>"
                                    }"""
            message1 = oci.generative_ai_inference.models.Message(
                role = "SYSTEM",
                content = [oci.generative_ai_inference.models.TextContent(
                    text = system_prompt,
                )]
            )
            message2 = oci.generative_ai_inference.models.Message(
                role="USER",
                content=[oci.generative_ai_inference.models.TextContent(
                    text=prompt
                )]
            )

            chat_request = oci.generative_ai_inference.models.GenericChatRequest(
                api_format=oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC,
                messages=[message1, message2],
                temperature=0.3,
                top_p=1

            )

            chat_details = ChatDetails(
                serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(model_id=self.oci_model_id),
                compartment_id=self.oci_compartment_id,
                chat_request=chat_request
            )

            # Call the OCI GenAI API
            response = self.oci_genai_client.chat(chat_details)
            # Assume response.data.choices[0].message.content is a JSON string as required

            analysis = json.loads(response.data.chat_response.choices[0].message.content[0].text)
            # print("Chat Response: \n", analysis)

            # Add metadata
            analysis['timestamp'] = datetime.now().isoformat()
            analysis['claim_id'] = claim_data.get('claim_id', 'N/A')
            analysis['model_used'] = 'oci-genai-gpt'
            analysis['model_version'] = self.oci_model_id

            # Normalize anomaly_label presentation
            anomaly_val = claim_data.get('anomaly_label', 0)
            if isinstance(anomaly_val, str) and anomaly_val.isdigit():
                anomaly_val = int(anomaly_val)
            elif isinstance(anomaly_val, bool):
                anomaly_val = int(anomaly_val)
            try:
                anomaly_val = int(anomaly_val)
            except Exception:
                anomaly_val = 0  # fallback

            analysis['anomaly_label'] = "Anomaly Detected" if anomaly_val == 1 else "No Anomaly"


            return analysis

        except Exception as e:
            return {
                'error': str(e),
                'fraud_score': 0,
                'risk_level': 'ERROR',
                'timestamp': datetime.now().isoformat()
            }


    
    def _create_fraud_analysis_prompt(self, claim_data: Dict):
        """Create a detailed prompt for fraud analysis"""
        
        prompt = f"""
        Analyze the following insurance claim for potential fraud:
        
        CLAIM DETAILS:
        - Claim ID: {claim_data.get('claim_id', 'N/A')}
        - Claim Type: {claim_data.get('claim_type', 'N/A')}
        - Claim Amount: ${claim_data.get('claim_amount', 0):,.2f}
        - Date of Incident: {claim_data.get('incident_date', 'N/A')}
        - Date of Claim Filing: {claim_data.get('filing_date', 'N/A')}
        
        POLICYHOLDER INFORMATION:
        - Policy Holder: {claim_data.get('policy_holder', 'N/A')}
        - Policy Number: {claim_data.get('policy_number', 'N/A')}
        - Policy Start Date: {claim_data.get('policy_start_date', 'N/A')}
        - Years as Customer: {claim_data.get('years_as_customer', 0)}
        
        INCIDENT DETAILS:
        - Location: {claim_data.get('incident_location', 'N/A')}
        - Description: {claim_data.get('incident_description', 'N/A')}
        - Police Report: {claim_data.get('police_report_filed', 'N/A')}
        - Summary: {claim_data.get('incident_summary', 'N/A')}
        
        DAMAGE DETAILS:
        - Evidence Provided: {claim_data.get('document_provided', 'N/A')}
        - Damage Type: {claim_data.get('damage_type', 'N/A')}     
        - Damage Description: {claim_data.get('damage_description', 'N/A')}
        
        ADDITIONAL CONTEXT:
        - Time Since Policy Start: {claim_data.get('days_since_policy_start', 0)} days
        - Claim Filing Delay: {claim_data.get('filing_delay_days', 0)} days
        
        ANOMALY DETECTION MODEL RESULT:
        - Anomaly Detection result: {claim_data.get('anomaly_label', False)}   
        
        Analyze this claim thoroughly and provide your assessment.
        """

        ##TODO: add claim.damage_type, claim.damage_detected, claim.damage_description


        # print("=== Claim Data Record ===\n")
        # print(prompt)
        # print("=========================\n")

        return prompt
    
    def batch_analyze_claims(self, claims: List[Dict]) -> List[Dict]:
        """
        Analyze multiple claims in batch
        
        Args:
            claims: List of claim dictionaries
            
        Returns:
            List of analysis results
        """
        results = []
        for claim in claims:
            analysis = self.analyze_claim(claim)
            results.append(analysis)
        return results
    
    def get_insights_summary(self, analyses: List[Dict]) -> Dict:
        """
        Generate summary insights from multiple analyses
        
        Args:
            analyses: List of analysis results
            
        Returns:
            Summary statistics and insights
        """
        if not analyses:
            return {}
        
        total = len(analyses)
        high_risk = sum(1 for a in analyses if a.get('risk_level') in ['HIGH', 'CRITICAL'])
        medium_risk = sum(1 for a in analyses if a.get('risk_level') == 'MEDIUM')
        low_risk = sum(1 for a in analyses if a.get('risk_level') == 'LOW')
        
        avg_fraud_score = sum(a.get('fraud_score', 0) for a in analyses) / total if total > 0 else 0
        avg_confidence = sum(a.get('confidence', 0) for a in analyses) / total if total > 0 else 0
        
        # Collect all fraud indicators
        all_indicators = []
        for analysis in analyses:
            all_indicators.extend(analysis.get('fraud_indicators', []))
        
        # Count frequency of indicators
        indicator_counts = {}
        for indicator in all_indicators:
            indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1
        
        # Get top indicators
        top_indicators = sorted(
            indicator_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return {
            'total_claims_analyzed': total,
            'high_risk_count': high_risk,
            'medium_risk_count': medium_risk,
            'low_risk_count': low_risk,
            'high_risk_percentage': (high_risk / total * 100) if total > 0 else 0,
            'average_fraud_score': round(avg_fraud_score, 2),
            'average_confidence': round(avg_confidence, 2),
            'top_fraud_indicators': [
                {'indicator': ind, 'frequency': count} 
                for ind, count in top_indicators
            ],
            'timestamp': datetime.now().isoformat()
        }


class NVIDIAFraudDetectionService:
    """
    Placeholder for NVIDIA fraud detection model integration
    Ready to integrate when inference server access is available
    """
    
    def __init__(self):
        self.endpoint = os.getenv('NVIDIA_INFERENCE_ENDPOINT', None)
        self.api_key = os.getenv('NVIDIA_API_KEY', None)
        self.is_available = False
    
    def check_availability(self) -> bool:
        """Check if NVIDIA inference server is available"""
        # TODO: Implement actual availability check
        return self.is_available
    
    def analyze_claim(self, claim_data: Dict) -> Dict:
        """
        Analyze claim using NVIDIA model
        Will be implemented when server access is available
        """
        if not self.is_available:
            return {
                'error': 'NVIDIA inference server not available',
                'fallback': 'Using OpenAI model'
            }
        
        # TODO: Implement NVIDIA model inference
        pass


class HybridFraudDetectionService:
    """
    Hybrid service that can use OpenAI or NVIDIA models
    Falls back to OpenAI if NVIDIA is unavailable
    """
    
    def __init__(self):
        self.openai_service = FraudDetectionService()
        self.nvidia_service = NVIDIAFraudDetectionService()
        self.prefer_nvidia = os.getenv('PREFER_NVIDIA', 'false').lower() == 'true'

    def analyze_claim(self, claim_data: Dict) -> Dict:
        """
        Analyze claim using available model
        Prefers NVIDIA if available and configured, falls back to OpenAI
        """
        if self.prefer_nvidia and self.nvidia_service.check_availability():
            result = self.nvidia_service.analyze_claim(claim_data)
            if 'error' not in result:
                return result
        
        # Use OpenAI as primary or fallback
        return self.openai_service.analyze_claim(claim_data)




if __name__ == "__main__":
    # Example usage
    service = FraudDetectionService()
    hybrid_service = HybridFraudDetectionService()
    
    # Sample claim data
    sample_claim = {
        'claim_id': 'CLM-2026-001',
        'claim_type': 'Auto Collision',
        'claim_amount': 15000,
        'incident_date': '2026-01-02',
        'filing_date': '2026-01-20',
        'policy_holder': 'John Doe',
        'policy_number': 'POL-123456',
        'policy_start_date': '2026-01-10',
        'previous_claims_count': 0,
        'years_as_customer': 0.1,
        'incident_location': 'Los Angeles, CA',
        'incident_description': 'Rear-end collision at intersection',
        'witnesses': 'None',
        'police_report_filed': 'No',
        'days_since_policy_start': 5,
        'filing_delay_days': 5,
        'similar_claims_in_area': 3,
        'repair_provider': 'Quick Fix Auto Body'
    }

    # print("Test OCI GenAI opt-oss-120b model")
    result = service.test_agent_hub()
    print("Analyzing claim with OpenAI...")
    # result = service.analyze_claim(sample_claim)
    # print(result)

    # inference_df = pd.read_csv("templates/sample_inference_data.csv")
    # print(service.claim_fraud_detect(inference_df.head(1)))

    # print(json.dumps(result, indent=2))

    # check HybridFraudDetection

    # result = hybrid_service.analyze_claim(sample_claim)
    # print(result)
