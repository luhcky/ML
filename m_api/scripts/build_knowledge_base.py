import os
import chromadb

# Must match main.py line 72-73 exactly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, ".chromadb")
COLLECTION_NAME = "mpesa_fraud_kb"

os.makedirs(DB_PATH, exist_ok=True)
client = chromadb.PersistentClient(path=DB_PATH)

# Use get_or_create, not get, and don't delete
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Only build if empty
if collection.count() == 0:
    documents = [
        "SIM Swap fraud: attacker tricks carrier to transfer victim's number to new SIM, receives OTPs, drains M-Pesa. Signals: new SIM + password reset + large withdrawal.",
        "Account Takeover: PIN stolen via phishing. Signals: new device login, off-hours, location change e.g. Rongai to Isiolo, high velocity.",
        "Social engineering: fake Safaricom call asking for PIN. Signals: reversal shortly after, small test then large amount.",
        "Till/Paybill fraud: fake supplier asks to pay new till like 90k. Signals: new beneficiary, round amount, first time.",
        "High risk rule: Account age <7 days and amount >50000 KES should be flagged, especially ASAL counties at night 10pm-5am.",
        "Velocity: >50 txns in 24h or 5 in 10min = bot/mule account. Require KYC.",
    ]
    collection.add(
        documents=documents,
        metadatas=[
            {"topic":"sim_swap"},
               {"topic":"account_take_over"},
               {"topic":"social_engineering"},
               {"topic":"till_fraud"},
               {"topic":"high_risk_value"},
               {"topic":"velocity"},
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    print(f"✅ Built {collection.count()} docs in {DB_PATH}/{COLLECTION_NAME}")
else:
    print(f"✅ Already exists: {collection.count()} docs")
