from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import detector
import time
import numpy as np
import cv2
import traceback
import threading

from detector import start_detector_thread, CURRENT_PREDICTION_DATA, CURRENT_FRAME, STATE_LOCK, FRAME_LOCK
from config import FLASK_HOST, FLASK_PORT, VIDEO_SOURCE, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER
from resources import RESOURCE_RECEIVERS
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

# --- MongoDB Setup ---
try:
    client_mongo = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    client_mongo.server_info()
    db = client_mongo.resq
    report_collection = db.report
except Exception as e:
    print("MongoDB connection failed:", e)
    report_collection = None

# --- Load T5 model ---
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# --- Start Detector Thread ---
start_detector_thread(
    video_source=VIDEO_SOURCE,
    model_dir="mobilenet",
    conf_threshold=0.5,
    inference_delay=0.08
)

# --- Twilio Client ---
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# --- Dispatch State ---
dispatch_state = {
    "status": "Not Sent",
    "timestamp": None,
    "receivers_map": {},
    "sids": {},
}

# --- Helper: Send WhatsApp with SMS fallback ---
def send_message_with_fallback(to_number, resource, incident_type, location, timestamp):
    body_text = (
        f"*RESQ ALERT*\nIncident: {incident_type}\nResource: {resource}\n"
        f"Location: {location}\nTime: {timestamp}\n\n"
        "Reply 'Confirm Dispatch' or 'Decline'."
    )
    # Try WhatsApp first
    try:
        msg = twilio_client.messages.create(
            from_='whatsapp:+18777804236',
            to=f'whatsapp:{to_number}',
            body=body_text
        )
        print(f"WhatsApp message sent to {to_number}, SID: {getattr(msg, 'sid', None)}")
        return getattr(msg, 'sid', None)
    except Exception as e:
        print(f"WhatsApp failed for {to_number}: {e}")
        # Fallback to SMS
        try:
            msg_sms = twilio_client.messages.create(
                from_=TWILIO_FROM_NUMBER,
                to=to_number,
                body=f"RESQ ALERT: Incident: {incident_type}, Resource: {resource}, Location: {location}, Time: {timestamp}"
            )
            print(f"SMS fallback sent to {to_number}, SID: {getattr(msg_sms, 'sid', None)}")
            return getattr(msg_sms, 'sid', None)
        except Exception as sms_e:
            print(f"SMS fallback also failed for {to_number}: {sms_e}")
            return None

# --- Logging ---
def log_incident(data):
    if report_collection is not None:
        doc = {
            "detected_objects": data.get('objects_detected', []),
            "objects_count": len(data.get('objects_detected', [])),
            "object_person_count": data.get('objects_detected', []).count('person'),
            "object_vehicle_count": data.get('objects_detected', []).count('vehicle'),
            "gps": data.get('location_gps', {'lat': 0, 'lng': 0}),
            "incident_type": data.get('incident_type', 'Unknown'),
            "multi_incident_string": data.get('events', ''),
            "report_text": data.get('final_report', ''),
            "timestamp": datetime.now(),
            "severity_level": data.get('severity_level', 3),
            "dispatch_status": data.get('dispatch_status', {}),
            "dispatch_state_snapshot": dispatch_state.copy(),
            "events": data.get('events', []),
            "resources_needed": data.get('resources_needed', [])
        }
        try:
            report_collection.insert_one(doc)
        except Exception as e:
            print("Failed to insert into MongoDB:", e)

# --- Allocate Resources ---
def allocate_resources(events):
    resources = set()
    for event in events:
        event_lower = event.strip().lower()
        for key, resource in {"fire": "Fire Truck", "jam": "Police", "person hit": "Ambulance", "crash": "Police"}.items():
            if key in event_lower:
                resources.add(resource)
    return list(resources)

# --- Core Dispatch (blocking) ---
def perform_dispatch(now_data):
    resources = now_data.get('resources_needed', [])
    if not resources:
        return False

    sids = {}
    all_receivers = []

    for resource in resources:
        receivers = RESOURCE_RECEIVERS.get(resource, [])
        all_receivers.extend(receivers)
        for number in receivers:
            sid = send_message_with_fallback(
                number,
                resource,
                now_data.get('incident_type', 'Unknown'),
                now_data.get('location_gps', 'Unknown'),
                now_data.get('timestamp', datetime.now().isoformat())
            )
            sids[number] = {"resource": resource, "sid": sid}

    with STATE_LOCK:
        dispatch_state['status'] = "Sent"
        dispatch_state['timestamp'] = datetime.now().isoformat()
        dispatch_state['receivers_map'] = {num: num for num in all_receivers}
        dispatch_state['sids'] = sids
        CURRENT_PREDICTION_DATA['dispatch_status'] = {
            num: {"status": "Sent", "resources": [sids[num]['resource']], "sid": sids[num]}
            for num in all_receivers
        }

    log_incident(CURRENT_PREDICTION_DATA)
    return True

# --- Core Dispatch (async wrapper) ---
def perform_dispatch_async(now_data):
    threading.Thread(target=perform_dispatch, args=(now_data,), daemon=True).start()

# --- Cancel Dispatch ---
def perform_cancel_dispatch():
    default_numbers = list(dispatch_state.get('receivers_map', {}).keys())
    if not default_numbers:
        return False

    cancel_sids = {}
    any_sent = False
    for number in default_numbers:
        try:
            msg_sid = send_message_with_fallback(
                number,
                "ALL",
                "Incident Cancelled",
                "N/A",
                datetime.now().isoformat()
            )
            cancel_sids[number] = msg_sid
            any_sent = True
        except Exception as e:
            print(f"Cancel message failed for {number}: {e}")
            cancel_sids[number] = None

    with STATE_LOCK:
        dispatch_state['status'] = "Cancelled" if any_sent else dispatch_state.get('status', 'Failed')
        dispatch_state['cancel_timestamp'] = datetime.now().isoformat()
        dispatch_state['cancel_sids'] = cancel_sids
        for num in default_numbers:
            CURRENT_PREDICTION_DATA['dispatch_status'][num] = {
                "status": "Cancelled",
                "resources": CURRENT_PREDICTION_DATA.get('resources_needed', [])
            }

    log_incident(CURRENT_PREDICTION_DATA)
    return any_sent

# --- Video Frame Generator (optimized) ---
def generate_frames():
    while True:
        with FRAME_LOCK:
            frame = detector.CURRENT_FRAME.copy() if detector.CURRENT_FRAME is not None else np.zeros((360, 640, 3), dtype=np.uint8)
        if frame is None:
            cv2.putText(frame, "CAMERA INITIALIZING...", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

# --- Flask Routes ---
@app.route('/')
def index():
    with STATE_LOCK:
        events = CURRENT_PREDICTION_DATA.get('events', '')
        events_list = [events] if isinstance(events, str) else events
        CURRENT_PREDICTION_DATA['resources_needed'] = allocate_resources(events_list)
    return render_template('index.html')

@app.route('/current_data')
def current_data():
    with STATE_LOCK:
        events = CURRENT_PREDICTION_DATA.get('events', '')
        events_list = [events] if isinstance(events, str) else events
        CURRENT_PREDICTION_DATA['resources_needed'] = allocate_resources(events_list)
        data = CURRENT_PREDICTION_DATA.copy()
        data['_dispatch_state'] = dispatch_state.copy()
    return jsonify(data)

@app.route('/auto_dispatch', methods=['POST'])
def auto_dispatch():
    with STATE_LOCK:
        data = CURRENT_PREDICTION_DATA.copy()
        perform_dispatch_async(data)  # Async call
    return jsonify({"status": "dispatched"})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/send_dispatch', methods=['POST'])
def send_dispatch():
    with STATE_LOCK:
        data = CURRENT_PREDICTION_DATA.copy()
    try:
        perform_dispatch_async(data)  # Async call
        return redirect(url_for('index'))
    except Exception as e:
        print("send_dispatch error:", e)
        return redirect(url_for('index'))

@app.route('/cancel_dispatch', methods=['POST'])
def cancel_dispatch():
    try:
        perform_cancel_dispatch()
    except Exception as e:
        print("cancel_dispatch error:", e)
    return redirect(url_for('index'))

# --- Twilio Webhook for WhatsApp ---
@app.route('/twilio_webhook', methods=['POST'])
def twilio_webhook():
    incoming_msg = request.form.get('Body', '').strip().lower()
    from_number = request.form.get('From', '').replace(' ', '').replace('-', '')
    response = MessagingResponse()

    with STATE_LOCK:
        dispatch_status_map = CURRENT_PREDICTION_DATA.get('dispatch_status', {})

        matched_number = None
        for num in dispatch_status_map.keys():
            if num[-10:] == from_number[-10:]:
                matched_number = num
                break

        if matched_number:
            user_status = dispatch_status_map[matched_number].get('status', 'Sent')
            resource = dispatch_status_map[matched_number]['resources'][0]

            if 'confirm' in incoming_msg and user_status == 'Sent':
                dispatch_status_map[matched_number]['status'] = 'Confirmed'
                response.message("Thank you. Your dispatch status has been logged.")

                # Notify others
                for num, info in dispatch_status_map.items():
                    if num != matched_number and info['resources'][0] == resource and info['status'] == 'Sent':
                        try:
                            send_message_with_fallback(num, resource, "No longer needed", "N/A", datetime.now().isoformat())
                            info['status'] = 'Cancelled'
                        except Exception as e:
                            print(f"Failed to notify {num}: {e}")

            elif 'decline' in incoming_msg and user_status == 'Sent':
                dispatch_status_map[matched_number]['status'] = 'Declined'
                response.message("You have declined the dispatch.")
            else:
                response.message("Your response cannot be processed. Dispatch already handled.")
        else:
            response.message("Your number is not recognized for any current dispatch.")

        CURRENT_PREDICTION_DATA['dispatch_status'] = dispatch_status_map

    log_incident(CURRENT_PREDICTION_DATA)
    return str(response)

@app.route('/receiver_location')
def receiver_location():
    with STATE_LOCK:
        loc = CURRENT_PREDICTION_DATA.get('receiver_location', {'lat': 0, 'lng': 0})
    return jsonify(loc)

@app.route('/history')
def history():
    incidents = []
    if report_collection is not None:
        incidents = list(report_collection.find().sort("timestamp", -1).limit(50))
    return render_template("history.html", incidents=incidents)

# --- Background Monitor (Single Incident) ---
def _dispatch_monitor_loop():
    print("Dispatch monitor started.")
    while True:
        try:
            with STATE_LOCK:
                now = CURRENT_PREDICTION_DATA.copy()
            incident_type = now.get('incident_type', 'Normal Flow')
            if incident_type and incident_type != "Normal Flow" and dispatch_state.get('status') != "Sent":
                perform_dispatch_async(now)  # Async call
            time.sleep(1.0)
        except Exception as e:
            print("Error in dispatch monitor loop:", e)
            traceback.print_exc()
            time.sleep(1.0)

monitor_thread = threading.Thread(target=_dispatch_monitor_loop, daemon=True)
monitor_thread.start()

if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True, use_reloader=False, threaded=True)
