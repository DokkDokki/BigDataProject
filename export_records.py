import os
import csv
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
uri = os.getenv("MONGO_URI")
if not uri:
    raise ValueError("MONGO_URI not found in .env")

client = MongoClient(uri)
client.admin.command("ping")
print("✅ MongoDB connected")

db = client["student_dropout_db"]
collection = db["records"]

docs = list(collection.find({}))
for d in docs:
    d["_id"] = str(d["_id"])

with open("records.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=docs[0].keys())
    writer.writeheader()
    writer.writerows(docs)

print("✅ Exported to records.csv")



