import os
import chromadb

# 1. Create client with persistent path (Render needs ./chroma_db)
os.makedirs("./chroma_db", exist_ok=True)
client = chromadb.PersistentClient(path="./chroma_db")

# 2. Delete old collection if exists (to avoid embedding mismatch)
try:
    client.delete_collection("fraud_kb")
except:
    pass

collection = client.get_or_create_collection(name="fraud_kb")

# 3. YOUR KNOWLEDGE BASE - add your fraud docs here
# You can expand this later
documents = [
    "SIM Swap fraud is when an attacker tricks a mobile carrier into transferring a victim's phone number to a new SIM card. They then receive OTPs and drain mobile money like M-Pesa. Signals: new SIM + password reset + large withdrawal within minutes.",
    "Account Takeover fraud: Attacker gains access to user's M-Pesa PIN via phishing. Signals: login from new device, off-hours transaction, change of location from e.g. Rongai to Isiolo, high velocity.",
    "Social engineering fraud: Victim is called by someone pretending to be Safaricom. They ask for PIN. Signals: victim initiates reversal shortly after transaction, small test amount then large amount.",
    "Business Email Compromise and Till fraud: Fraudster pretends to be supplier and asks to pay to new till number. Signals: new beneficiary, round amount like 90,000, first time transaction.",
    "New account + large amount is highest risk for mobile money fraud. Account age < 7 days and amount > 50000 KES should always be flagged for review, especially if from ASAL region or at night 10pm-5am.",
    "Velocity fraud: More than 10 transactions in 24h or 3 transactions in 10 minutes indicates bot or mule account. Block and require KYC.",
]

ids = [f"doc_{i}" for i in range(len(documents))]

collection.add(
    documents=documents,
    ids=ids
)

print(f"✅ RAG knowledge base built! Added {len(collection.get()['ids'])} docs to ./chroma_db")