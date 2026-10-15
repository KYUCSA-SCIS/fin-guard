import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# LangChain & Vector Store Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


print("1. Generating synthetic transaction dataset...")
np.random.seed(42)
n_samples = 5000

transaction_id = [f"TXN_{100000 + i}" for i in range(n_samples)]
amount_ugx = np.clip(np.random.exponential(scale=150000, size=n_samples).round(-3), 2000, 25000000)
sim_swap_age_hours = np.random.exponential(scale=120, size=n_samples)
recipient_wallet_age_days = np.random.exponential(scale=180, size=n_samples)
p2p_velocity_1hr = np.random.poisson(lam=1.2, size=n_samples)
distance_from_home_km = np.random.exponential(scale=15, size=n_samples)
device_imei_match = np.random.choice([1, 0], size=n_samples, p=[0.92, 0.08])

df = pd.DataFrame({
    'transaction_id': transaction_id,
    'amount_ugx': amount_ugx,
    'sim_swap_age_hours': sim_swap_age_hours.round(1),
    'recipient_wallet_age_days': recipient_wallet_age_days.round(1),
    'p2p_velocity_1hr': p2p_velocity_1hr,
    'distance_from_home_km': distance_from_home_km.round(2),
    'device_imei_match': device_imei_match,
    'is_fraud': 0
})

# Inject local fraud patterns
fraud_condition_A = (df['sim_swap_age_hours'] < 24) & (df['recipient_wallet_age_days'] < 3)
fraud_condition_B = (df['device_imei_match'] == 0) & (df['p2p_velocity_1hr'] > 5) & (df['distance_from_home_km'] > 80)
df.loc[fraud_condition_A | fraud_condition_B, 'is_fraud'] = 1

# Add noise/stealth fraud
stealth_idx = df[df['is_fraud'] == 0].sample(frac=0.015, random_state=42).index
df.loc[stealth_idx, 'is_fraud'] = 1


print("2. Training Random Forest Fraud Classifier...")
X = df.drop(columns=['transaction_id', 'is_fraud'])
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)


print("3. Indexing Bank of Uganda Regulatory Knowledge Base into ChromaDB...")
bou_legal_text = """
BANK OF UGANDA FINANCIAL CONSUMER PROTECTION GUIDELINES (2011)

Section 8: Disclosure and Transparency Requirements
(1) Financial service providers shall disclose all terms, conditions, fees, and interest rates to consumers in plain, simple, and understandable language.
(2) In cases of suspected fraud or unauthorized account access, the provider may place a temporary hold on outward transfers for up to 24 hours.
(3) When a hold is applied, the provider must immediately issue a plain-language notification to the consumer via SMS or in-app prompt stating the reason for the hold and providing instructions on how to lodge a dispute or verify identity.

Section 11: Redress and Complaints Handling
(1) Consumers have the right to challenge any automated transaction hold or account restriction.
(2) The financial service provider must assign a unique tracking ticket ID to every dispute and resolve or provide a formal update within 14 business days.
(3) Universal Service Obligations require providers to preserve access to emergency contacts and basic utility payment channels even during account restriction states.
"""

# Save regulatory text locally
txt_filename = "bou_fcp_2011.txt"
with open(txt_filename, "w") as f:
    f.write(bou_legal_text)

# Chunk and Vectorize
loader = TextLoader(txt_filename)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "./chroma_bou_db"

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_directory
)


def query_policy_rag(risk_type: str, query_text: str):
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    results = db.similarity_search(query_text, k=1)
    retrieved_clauses = [doc.page_content for doc in results]
    return retrieved_clauses


def run_finguard_pipeline(transaction_data):
    # Calculate Fraud Probability
    risk_score = clf.predict_proba(transaction_data)[:, 1][0]
    print(f"\n[EVALUATION] Calculated Risk Score: {risk_score * 100:.1f}%")

    if risk_score >= 0.75:
        # High-Risk Interception via RAG Lookups
        legal_context = query_policy_rag(
            risk_type="High Fraud Risk",
            query_text="Rules for Accounts Restrictions and temporary transaction holds"
        )

        backing_clause = legal_context[0] if legal_context else "Bank of Uganda Financial Consumer Protection Guidelines (2011) Section 8."

        action_payload = {
            "status": "INTERCEPTED_PROPORTIONAL_HOLD",
            "risk_score": f"{risk_score * 100:.1f}%",
            "enforced_action": "Applied 24-Hour Temporary Outward Hold (Preserved Emergency Lines)",
            "consumer_sms": "Warning: Suspected unauthorized transaction. A temporary 24-hr hold has been placed on outward transfers. Reply '1' or click here to verify identity or lodge a dispute.",
            "regulatory_backing": backing_clause
        }
        return action_payload
    else:
        # Normal/Low-Risk Settlement
        action_payload = {
            "status": "SETTLED",
            "risk_score": f"{risk_score * 100:.1f}%",
            "enforced_action": "Instant Transaction Settlement Approved",
            "consumer_sms": "Transaction successful.",
            "regulatory_backing": "Complies with standard National Payment Systems Act (2020) settlement rules."
        }
        return action_payload


if __name__ == "__main__":
    print("\n=======================================================")
    print("Testing FinGuard AI Pipeline with Sample Inputs")
    print("=======================================================")

    # Test Case 1: High Risk Scenario (SIM Swap < 3 hrs + Unrecognized IMEI + High Velocity)
    sample_fraud = pd.DataFrame([{
        'amount_ugx': 5000000,
        'sim_swap_age_hours': 2.5,
        'recipient_wallet_age_days': 1.0,
        'p2p_velocity_1hr': 6,
        'distance_from_home_km': 120.0,
        'device_imei_match': 0
    }])

    # Test Case 2: Normal/Safe Scenario (Old SIM + Recognized Device + Normal Velocity)
    sample_safe = pd.DataFrame([{
        'amount_ugx': 50000,
        'sim_swap_age_hours': 720.0,
        'recipient_wallet_age_days': 300.0,
        'p2p_velocity_1hr': 1,
        'distance_from_home_km': 2.0,
        'device_imei_match': 1
    }])

    print("\n--- TEST CASE 1: SUSPECTED FRAUDULENT TRANSFER ---")
    result_1 = run_finguard_pipeline(sample_fraud)
    for k, v in result_1.items():
        print(f"{k.upper()}: {v}")

    print("\n--- TEST CASE 2: NORMAL P2P TRANSFER ---")
    result_2 = run_finguard_pipeline(sample_safe)
    for k, v in result_2.items():
        print(f"{k.upper()}: {v}")