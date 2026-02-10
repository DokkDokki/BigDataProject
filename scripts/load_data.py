import os
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi

load_dotenv()

uri = os.getenv("MONGO_URI")

client = MongoClient(
    uri,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=5000
)

client.admin.command("ping")
print("✅ Ping OK")

db = client["student_dropout_db"]
print("Collections:", db.list_collection_names())



