from pymongo import MongoClient
from dotenv import load_dotenv
import os
import csv

# 1) โหลดค่าจาก .env
load_dotenv()
uri = os.getenv("MONGO_URI")

# 2) เชื่อม MongoDB
client = MongoClient(uri)
db = client["student_dropout_db"]
collection = db["predictions"]

# 3) ดึงข้อมูลทั้งหมด
docs = list(collection.find({}))

if len(docs) == 0:
    print("❌ predictions collection is empty")
    exit()

# 4) เขียน CSV
with open("predictions.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = docs[0].keys()
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for doc in docs:
        doc["_id"] = str(doc["_id"])
        writer.writerow(doc)

print("✅ Exported to predictions.csv")
