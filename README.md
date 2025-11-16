# 🚨 Real-Time Road Incident/Hazard Detection and SMS Alert Notification System

This system integrates **computer vision**, **AI report generation**, **Flask backend**, and **Twilio SMS dispatch** to deliver a complete real-time hazard detection + emergency alerting workflow.

---

# 📖 TABLE OF CONTENTS

* <b>[📝 Overview](#overview)</b>
    * [🧩 Core Features](#core-features)
    * [💥 Incidents the CV/YOLO Model Detects](#incidents-the-cvyolo-model-detects)

* <b>[📹 Project Demo](#project-demo)</b>

* <b>[⚙️ Technology Stack](#technology-stack)</b>

* <b>[🔄 Workflow](#workflow)</b>
    * [🖧 Diagram](#diagram)
    * [📝 Steps](#steps)

* <b>[📁 Project Structure](#project-structure)</b>
    * [🐍 Root Python Modules](#root-python-modules)
    * [📹 Mobilenet Model Folder](#model-folder)
    * [🎨 UI Layer](#ui-layer)
    * [🧪 Dependencies](#dependencies)

* <b>[🔧 Configuration Guide](#configuration-guide)</b>
    * [💬 Create a Free Twilio Account](#step-1---create-a-free-twilio-account)
    * [⛁ Create a Free MongoDB Atlas Account](#step-2---create-a-free-mongodb-atlas-account)

* <b>[🚀 Steps to Run The Application](#steps-to-run-the-application)</b>

* <b>[✅ Final Notes](#final-notes)</b>

* <b>[License (MIT)](#license)</b>

---

## Overview

This system continuously monitors video feeds, detects hazards, generates natural-language incident reports using T5, and alerts responders through SMS.

### Core Features

* 📹 **Real-time video analysis** (OpenCV + MobileNet SSD + optional YOLO)
* 🧠 **Automatic incident reports** using a T5 model
* 🕹️ **Controller verification interface**
* ✉️ **SMS/WhatsApp dispatch via Twilio**
* 📱 **Mobile App Notifications**
* 🌐 **Flask Dashboard** for real-time monitoring

### Incidents the CV/YOLO Model Detects

The YOLO model is pretrained to detect a variety of objects, such as cars, buses, autorickshaws, buildings, trees, and dogs and cats.
* Vehicle Crash 💥 (vehicle bounding boxes touching/overlapping logic)
* Traffic Jam 🚧 (proximity and clustering logic)
* Fire 🔥 (HSV color detection logic)
* Person Hit by Vehicle 💥 (approximate proximity logic)

## Project Demo

For our hackathon demo, we used toy cars to simulate a real car crash.

<img width="1389" height="617" alt="Screenshot from 2025-11-16 13-43-00" src="https://github.com/user-attachments/assets/79ffb09b-3f4b-4fa7-9426-e50f9181bf51">

After an incident is detected (in this case, fire), an SMS alert is triggered via Twilio API.

<img width=auto height="500" alt="Screenshot from 2025-10-26 06-03-47" src="https://github.com/user-attachments/assets/10649a17-e1d8-44d5-a2b1-dbffad3adaa0" />

---

## Technology Stack

### Web & Backend Frameworks
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask)
![Gunicorn](https://img.shields.io/badge/Gunicorn-499848?style=for-the-badge&logo=gunicorn)

**- flask**: Web server, routes, dashboard, controller UI  
**- gunicorn**: Production WSGI server (deployment)

### Computer Vision (CV) Stack
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-blue?style=for-the-badge)

**- opencv-python**: OpenCV with GUI support (local testing)  
**- opencv-python-headless**: Headless OpenCV for servers / Docker  
**- ultralytics**: YOLOv8 (optional real-time object detection)

### Deep Learning / AI Components
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas)

**- torch**: PyTorch backend for T5 + model inference  
**- transformers**: HuggingFace Transformers (T5 model)  
**- sentencepiece**: Tokenizer dependency for T5  
**- tokenizers**: Fast tokenizer library (HF)  
**- scikit-learn**: ML utilities (clustering logic, metrics)  
**- datasets**: HuggingFace dataset loader (optional)  
**- pandas**: Data handling for logs and tables

### Database Layer
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb)

**- pymongo[srv]**: MongoDB Atlas connector (SRV protocol)

### Messaging / API Integrations
![Twilio](https://img.shields.io/badge/Twilio-F22F46?style=for-the-badge&logo=twilio)

**- twilio**: SMS notifications API

### Utilities (Optional)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib)

**- numpy**: Numerical operations used across CV + ML  
**- matplotlib**: Debug visuals, heatmaps, plotting

---

## Workflow

### Diagram

```
Camera → OpenCV Detection → Incident? → T5 Report → Twilio SMS Dispatch
```

### Steps

1. 📡 Capture video feed.
2. 🧠 Run CV incident detection.
3. 📝 Generate AI narrative.
4. 📤 SMS/WhatsApp dispatch.
5. 🚑 Responders notified (Ambulance/Police/Fire Station).

---

## Project Structure

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

## Root Python Modules

### `app.py` — Flask Backend & Dispatch Engine

Manages web dashboard, routing, video feed, T5 pipeline, Twilio SMS, MongoDB logging, and resource allocation.

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

## Model Folder

MobileNet SSD pretrained model files.

---

## UI Layer

### `index.html`
Dashboard (video feed + real-time generated incident data)
### `history.html`
Log history (fetched dynamically from MongoDB)

---

## Dependencies

### `requirements.txt`

```
## --- Web & Backend Frameworks --- ##
flask                 # Web server, routes, dashboard, controller UI
gunicorn              # Production WSGI server (deployment)


## --- Computer Vision (CV) Stack --- ##
opencv-python         # OpenCV with GUI support (local testing)
opencv-python-headless # Headless OpenCV for servers / Docker
ultralytics           # YOLOv8 (optional real-time object detection)


## --- Deep Learning / AI Components --- ##
torch                 # PyTorch backend for T5 + model inference
transformers          # HuggingFace Transformers (T5 model)
sentencepiece         # Tokenizer dependency for T5
tokenizers            # Fast tokenizer library (HF)
scikit-learn          # ML utilities (clustering logic for jams, metrics)
datasets              # HuggingFace dataset loader (optional)
pandas                # Data handling for logs and tables


## --- Database Layer --- ##
pymongo[srv]          # MongoDB Atlas connector (SRV protocol)


## --- Messaging / API Integrations --- ##
twilio                # SMS / WhatsApp notifications API


## --- Utilities (Optional) --- ##
numpy                 # Numerical operations used across CV + ML
matplotlib            # Debug visuals, heatmaps, plotting
```

---

## Configuration Guide

Below are **all setup steps** required before running the application.

## Step 1 - Create a Free Twilio Account

**URL:** [SMS Messaging, Built to Scale](https://login.twilio.com/u/signup?state=hKFo2SBPN0VrQ0R3LTBaQTBKRG5MdTZWVjZmdjlQbThwWGxmTKFur3VuaXZlcnNhbC1sb2dpbqN0aWTZIDdQUkRqTjFPVHY5VlRQM09Ncm1naWFaaUFTTXYtU0FCo2NpZNkgTW05M1lTTDVSclpmNzdobUlKZFI3QktZYjZPOXV1cks) 💬

Sign up for a free account → verify phone → get a **Twilio phone number**.

You will need:

* **Account SID**
* **Auth Token**
* **Twilio phone number** (formatted like `+15551234567`)

You will insert these into `config.py`.

## Step 2 - Create a Free MongoDB Atlas Account

**URL:** [https://www.mongodb.com/cloud/atlas/register](https://www.mongodb.com/cloud/atlas/register)

Create a cluster → create a database → create a collection.

You need:

* **MongoDB URI** (e.g. `mongodb+srv://username:password@cluster.mongodb.net/`)
* **Database Name** ('resq')
* **Collection Name** ('reports')

You will insert these into `config.py`.

---

### Your `config.py` (Insert Your Details)

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

#### Replace with:

* `your_account_sid` → From Twilio Dashboard
* `your_auth_token` → From Twilio Dashboard
* `your_from_number` → Your Twilio phone number

* `your_uri` → MongoDB connection string
* `"resq"` → Database name (or change)
* `"reports"` → Your collection name

---

### Your `resources.py` (Insert Your Details)

```python
# Mapping of resource type to default receiver phone numbers
RESOURCE_RECEIVERS = {
    "Ambulance": ["no."],
    "Fire Truck": ["no.", "no."],
    "Police": ["no.", "no."],
}
```

#### Replace:

* `"no."` → Real phone numbers in **international format**, e.g.:

  ```
  "+15551234567"
  ```

You can add multiple numbers per emergency unit.

---

## Steps to Run The Application

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
cd Road-Incident-Hazard-Detection-and-SMS-Alert-Notification-System
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

Then open the url:

```
http://127.0.0.1:5000
```

---

## Final Notes

* Never commit secrets like your MongoDB or Twilio account passwords or phone numbers.
* Use environment variables in production.
* Ensure webcam or RTSP feed is configured on your PC.

---

## License

This project is under the MIT license and is free to copy, modify, and distribute with appropriate attribution. Read the license at [LICENSE.md](LICENSE.md)
