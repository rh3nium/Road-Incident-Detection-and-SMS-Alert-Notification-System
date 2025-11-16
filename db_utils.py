# db_utils.py
from pymongo import MongoClient
from config import MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION

_client = None
_db = None
_collection = None

def _get_collection():
    global _client, _db, _collection
    if _collection is None:
        _client = MongoClient(MONGO_URI)
        _db = _client[MONGO_DB_NAME]
        _collection = _db[MONGO_COLLECTION]
    return _collection

def log_report_to_db(report_doc):
    coll = _get_collection()
    coll.insert_one(report_doc)
