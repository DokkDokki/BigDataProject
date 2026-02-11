import os
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi
import pandas as pd

def load_data(): # Connect to MongoDB and load Data

    load_dotenv()

    uri = os.getenv("MONGO_URI")

    client = MongoClient(
        uri,
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=5000
    )

    client.admin.command("ping")
    print("✅ MongoDB Connection OK")

    db = client["student_dropout_db"]
    collections = db.list_collection_names()
    print("Collections:", collections)

    if not collections:
        raise Exception("❌ No collections found in MongoDB database.")

    collection = db[collections[0]]  
    data = list(collection.find({}, {"_id": 0}))  # Remove _id

    df = pd.DataFrame(data) # Create DataFrame

    print("📄 Loaded DataFrame:", df.shape)
    return df


if __name__ == "__main__":
    # This runs ONLY if you click "Run" on load_data.py
    df = load_data()
    print(df.head())
