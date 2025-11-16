# config.py
# Secret keys / URIs. DO NOT COMMIT with real credentials to public repos.

# Twilio credentials
TWILIO_ACCOUNT_SID = "your_account_sid"
TWILIO_AUTH_TOKEN = "your_auth_token"
TWILIO_FROM_NUMBER = "your_from_number"  # Twilio virtual number

# MongoDB (Atlas) connection string (use srv or non-srv)
MONGO_URI = "your_uri"
MONGO_DB_NAME = "resq"
MONGO_COLLECTION = "reports"

# T5 Model ('t5-small' or path to local checkpoint)
T5_MODEL_NAME = "t5-small"

# Video source:
VIDEO_SOURCE = 0 # local webcam
# "http://<ip>:8080/video" or RTSP URL

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
