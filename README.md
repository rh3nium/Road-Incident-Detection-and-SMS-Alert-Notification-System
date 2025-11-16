# 🚨 Real-Time Road Incident/Hazard Detection and SMS Alert Notification System

This system integrates **computer vision**, **AI report generation**, **Flask backend**, and **Twilio SMS dispatch** to deliver a complete real-time hazard detection + emergency alerting workflow.

---

# 📖 TABLE OF CONTENTS

* [⚙️ Overview](#%EF%B8%8F-overview)
    * [🧩 Core Features](#%F0%9F%A7%A9-core-features)
    * [💥 Incidents the CV/YOLO Model Detects](#%F0%9F%92%A5-incidents-the-cvyolo-model-detects)
* [🔄 Workflow](#%F0%9F%94%81-workflow)
    * [📝 Steps](#%F0%9F%93%8D-steps)
* [📁 Project Structure](#%F0%9F%93%81-project-structure)
    * [🐍 Root Python Modules](#%F0%9F%90%8D-root-python-modules)
    * [📁 Model Folder](#%F0%9F%93%81-model-folder)
    * [🎨 UI Layer](#%F0%9F%8E%A8-ui-layer)
    * [🧪 Dependencies](#%F0%9F%A7%AA-dependencies)
* [🔧 Configuration Guide](#%F0%9F%94%A7-configuration-guide)
    * [💬 Step 1 — Create a Free Twilio Account](#%F0%9F%97%83%EF%B8%8F-step-1--create-a-free-twilio-account)
    * [🛢️ Step 2 — Create a Free MongoDB Atlas Account](#%F0%9F%9B%B2%EF%B8%8F-step-2--create-a-free-mongodb-atlas-account)
* [🚀 Steps to Run The Application](#%F0%9F%9A%80-steps-to-run-the-application)
* [✔ Final Notes](#%E2%9C%94-final-notes)

---

## ⚙️ Overview

This system continuously monitors video feeds, detects hazards, generates natural-language incident reports using T5, and alerts responders through SMS/WhatsApp.

### 🧩 Core Features

* 📹 **Real-time video analysis** (OpenCV + MobileNet SSD + optional YOLO)
* 🧠 **Automatic incident reports** using a T5 model
* 🕹️ **Controller verification interface**
* ✉️ **SMS/WhatsApp dispatch via Twilio**
* 📱 **Mobile App Notifications**
* 🌐 **Flask Dashboard** for real-time monitoring

### 💥 Incidents the CV/YOLO Model Detects

* Vehicle Crash
* Traffic Jam
* Fire
* Person Hit by Vehicle (approximate proximity logic)

For our hackathon demo, we used toy cars to simulate a real car crash.



---

## 🔄 Workflow

```
Camera → OpenCV Detection → Incident? → T5 Report → Twilio SMS Dispatch
```

### 📝 Steps

1. 📡 Capture video feed.
2. 🧠 Run CV incident detection.
3. 📝 Generate AI narrative.
4. 🧑‍✈️ Controller verifies.
5. 📤 SMS/WhatsApp dispatch.
6. 🚑 Responders notified.

---

## 📁 Project Structure

```
resq/
├── app.py
├── main.py
├── detector.py
├── t5_generator.py
├── db_utils.py
├── sms_utils.py
├── resources.py
├── config.py
├── requirements.txt
├── mobilenet/
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
├── templates/
│   ├── index.html
│   └── history.html
└── static/
```

---

## 🐍 Root Python Modules

### `app.py` — Flask Backend & Dispatch Engine

Manages routing, video feed, T5 pipeline, Twilio SMS, MongoDB logging, and resource allocation.

### `detector.py` — Real-time CV Pipeline

Handles MobileNet SSD detection, fire detection, crash logic, jam clustering, bounding boxes, and AI report triggering.

### `main.py` — Incident Classification

Implements clustering, overlap detection, event merging, and priority scoring.

### `t5_generator.py`

Produces narrative incident reports and follow-up event predictions.

### `db_utils.py`

MongoDB logging utilities.

### `sms_utils.py`

Minimal Twilio SMS sender.

### `resources.py`

Defines emergency receiver numbers.

### `config.py`

Contains Twilio/MongoDB credentials, model name, server settings.

---

## 📁 Model Folder

MobileNet SSD pretrained model files.

---

## 🎨 UI Layer

`index.html` — dashboard (video feed + logs)
`history.html` — log history (MongoDB)

---

## 🧪 Dependencies

### `requirements.txt`

### Web & Backend Frameworks
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask)
![Gunicorn](https://img.shields.io/badge/Gunicorn-499848?style=for-the-badge&logo=gunicorn)

flask                 # Web server, routes, dashboard, controller UI  
gunicorn              # Production WSGI server (deployment)

### Computer Vision (CV) Stack
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-blue?style=for-the-badge)

opencv-python          # OpenCV with GUI support (local testing)  
opencv-python-headless # Headless OpenCV for servers / Docker  
ultralytics            # YOLOv8 (optional real-time object detection)

### Deep Learning / AI Components
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas)

torch                 # PyTorch backend for T5 + model inference  
transformers          # HuggingFace Transformers (T5 model)  
sentencepiece         # Tokenizer dependency for T5  
tokenizers            # Fast tokenizer library (HF)  
scikit-learn          # ML utilities (clustering logic, metrics)  
datasets              # HuggingFace dataset loader (optional)  
pandas                # Data handling for logs and tables

### Database Layer
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb)

pymongo[srv]          # MongoDB Atlas connector (SRV protocol)

### Messaging / API Integrations
![Twilio](https://img.shields.io/badge/Twilio-F22F46?style=for-the-badge&logo=twilio)

twilio                # SMS / WhatsApp notifications API

### Utilities (Optional)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib)

numpy                 # Numerical operations used across CV + ML  
matplotlib            # Debug visuals, heatmaps, plotting

---

## 🔧 Configuration Guide

Below are **all setup steps** required before running the application.

---

## 💬 Step 1 — Create a Free Twilio Account

**URL:** [https://www.twilio.com/try-twilio](https://www.twilio.com/try-twilio)

Sign up for a free account → verify phone → get a **Twilio phone number**.

You will need:

* **Account SID**
* **Auth Token**
* **Twilio phone number** (formatted like `+15551234567`)

You will insert these into `config.py`.

---

## 🛢️ Step 2 — Create a Free MongoDB Atlas Account

**URL:** [https://www.mongodb.com/cloud/atlas/register](https://www.mongodb.com/cloud/atlas/register)

Create a cluster → create a database → create a collection.

You need:

* **MongoDB URI** (e.g. `mongodb+srv://username:password@cluster.mongodb.net/`)
* **Database Name** (default: `resq`)
* **Collection Name** (default: `reports`)

You will insert these into `config.py`.

---

## Your `config.py` (Insert Your Details)

```python
# Twilio credentials
TWILIO_ACCOUNT_SID = "your_account_sid"
TWILIO_AUTH_TOKEN = "your_auth_token"
TWILIO_FROM_NUMBER = "your_from_number"

# MongoDB settings
MONGO_URI = "your_uri"
MONGO_DB_NAME = "resq"
MONGO_COLLECTION = "reports"

# T5 model
T5_MODEL_NAME = "t5-small"

# Video source
VIDEO_SOURCE = 0

# Flask
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
```

### Replace with:

* `your_account_sid` → From Twilio Dashboard
* `your_auth_token` → From Twilio Dashboard
* `your_from_number` → Your Twilio phone number

* `your_uri` → MongoDB connection string
* `"resq"` → Database name (or change)
* `"reports"` → Your collection name

---

## Your `resources.py` (Insert Your Details)

```python
# Mapping of resource type to default receiver phone numbers
RESOURCE_RECEIVERS = {
    "Ambulance": ["no."],
    "Fire Truck": ["no.", "no."],
    "Police": ["no.", "no."],
}
```

### Replace:

* `"no."` → Real phone numbers in **international format**, e.g.:

  ```
  "+15551234567"
  ```

You can add multiple numbers per emergency unit.

---

## 🚀 Steps to Run The Application

### 🔹 1. Navigate to Desired Location

Use the `cd` command to move to the directory where you want the project stored:

```
cd ~/your/desired/path
```

---

### 🔹 2. Clone the Repository

```
git clone https://github.com/rh3nium/Road-Incident-Hazard-Detection-and-SMS-Alert-Notification-System/tree/main
```

---

### 🔹 3. Navigate to the Repository Folder

```
cd resq
```

---

### 🔹 4. Create Virtual Environment

```
python3 -m venv env
```

### Activate:

```
source env/bin/activate
```

---

### 🔹 5. Install Dependencies

```
pip install -r requirements.txt
```

---

### 🔹 6. Run the Flask Server

```
python3 app.py
```

Then open:

```
http://127.0.0.1:5000
```

---

## ✔ Final Notes

* Never commit real passwords.
* Use environment variables in production.
* Ensure webcam or RTSP feed is configured.
