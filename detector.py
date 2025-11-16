import time
import threading
import numpy as np
import cv2
from t5_generator import generate_report_from_incident
from config import VIDEO_SOURCE

STATE_LOCK = threading.Lock()
FRAME_LOCK = threading.Lock()

CURRENT_PREDICTION_DATA = {
    'incident_type': 'Initializing...',
    'location_gps': 'N/A',
    'timestamp': 'N/A',
    'objects_detected': [],
    'narrative_resources': 'Please wait for initialization...',
    'events': 'N/A',
    'final_report': 'N/A',
    'resources_needed': [],
    'active_incidents': []
}

CURRENT_FRAME = None

INCIDENT_TO_RESOURCES = {
    "fire": ["Fire Truck"],
    "jam": ["Police"],
    "person hit": ["Ambulance"],
    "crash": ["Police"]
}

def get_dynamic_metadata():
    gps = "PES University RR Campus, Banashankari, Bengaluru 560085"
    timestamp = time.strftime("%H:%M:%S, %d %b %Y", time.localtime())
    return gps, timestamp

def start_detector_thread(video_source=VIDEO_SOURCE, model_dir="mobilenet", conf_threshold=0.5, inference_delay=0.08, use_yolo=False):
    thread = threading.Thread(
        target=_detector_worker,
        args=(video_source, model_dir, conf_threshold, inference_delay, use_yolo),
        daemon=True
    )
    thread.start()
    return thread

def _detector_worker(video_source, model_dir, conf_threshold, inference_delay, use_yolo):
    global CURRENT_FRAME, CURRENT_PREDICTION_DATA
    from main import classify_incident, VEHICLE_CLASSES, OBSTACLE_CLASSES

    # --- Load MobileNet SSD ---
    try:
        prototxt = f"{model_dir}/MobileNetSSD_deploy.prototxt"
        caffemodel = f"{model_dir}/MobileNetSSD_deploy.caffemodel"
        net_mobilenet = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        net_mobilenet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception as e:
        print("Error loading MobileNet model:", e)
        return

    # --- Load YOLOv8 (optional) ---
    net_yolo = None
    if use_yolo:
        try:
            from ultralytics import YOLO
            net_yolo = YOLO("yolov8n.pt")
        except Exception as e:
            print("YOLOv8 not available:", e)
            use_yolo = False

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Video source {video_source} not available. Exiting detector.")
        return

    MOBILENET_CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
        "tvmonitor", "truck"
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue

        h, w = frame.shape[:2]
        detected_objects = []
        vehicle_boxes = []
        obstacle_boxes = []

        # --- MobileNet Detection ---
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net_mobilenet.setInput(blob)
        detections = net_mobilenet.forward()
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            idx = int(detections[0, 0, i, 1])
            if idx < 0 or idx >= len(MOBILENET_CLASSES):
                continue
            class_name = MOBILENET_CLASSES[idx]
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            bbox_norm = [x1 / w, y1 / h, x2 / w, y2 / h]
            detected_objects.append(class_name)
            if class_name in VEHICLE_CLASSES or class_name == 'person':
                vehicle_boxes.append({'box': bbox_norm, 'class': class_name})
            if class_name in OBSTACLE_CLASSES:
                obstacle_boxes.append(bbox_norm)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {int(conf * 100)}%", (max(0, x1), max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # --- YOLOv8 Detection ---
        if use_yolo and net_yolo is not None:
            try:
                results = net_yolo(frame, conf=0.4)[0]
                for r in results.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, cls = r
                    class_name = net_yolo.model.names[int(cls)]
                    detected_objects.append(class_name)
                    if class_name in VEHICLE_CLASSES or class_name == 'person':
                        vehicle_boxes.append({'box': [x1 / w, y1 / h, x2 / w, y2 / h], 'class': class_name})
                    if class_name in OBSTACLE_CLASSES:
                        obstacle_boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {int(conf * 100)}%", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            except Exception as e:
                print("YOLO detection error:", e)

        # --- Fire detection ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_fire = np.array([10, 150, 150])
        upper_fire = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_fire, upper_fire)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fire_boxes = []
        for c in contours:
            if cv2.contourArea(c) > 400:
                x, y, w_box, h_box = cv2.boundingRect(c)
                fire_boxes.append([x, y, x + w_box, y + h_box])
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)

        # --- Multi-incident classification ---
        if 'active_incidents' not in CURRENT_PREDICTION_DATA:
            CURRENT_PREDICTION_DATA['active_incidents'] = []

        keywords = classify_incident(
            detected_objects, vehicle_boxes, obstacle_boxes, fire_boxes,
            active_incidents=CURRENT_PREDICTION_DATA.get('active_incidents', [])
        )

        gps, timestamp = get_dynamic_metadata()

        # --- Generate SEPARATE reports for each detected incident ---
        reports = []
        for inc in keywords.get('active_incidents', []):
            try:
                single_report = generate_report_from_incident({
                    "incident_type": inc['type'],
                    "objects_detected": detected_objects,
                    "multi_incident_string": f"{inc['type']} (P{inc['priority']})"
                })
                reports.append(single_report)
            except Exception as e:
                print("Report generation error for", inc['type'], ":", e)

        report_text = "\n\n".join(reports) if reports else "Report generation failed."

        # --- Determine required resources ---
        resources_needed = set()
        for inc in keywords['active_incidents']:
            resources_needed.update(INCIDENT_TO_RESOURCES.get(inc['type'].lower(), []))

        status_text = keywords.get('incident_type', 'N/A')
        color = (0, 0, 255) if status_text.lower() != 'normal flow' else (0, 255, 0)
        cv2.putText(frame, f"EVENT: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # --- Update shared prediction data ---
        with STATE_LOCK:
            CURRENT_PREDICTION_DATA.update({
                'incident_type': keywords.get('incident_type', 'N/A'),
                'location_gps': gps,
                'timestamp': timestamp,
                'objects_detected': detected_objects,
                'narrative_resources': report_text,
                'events': keywords.get('multi_incident_string', ''),
                'final_report': report_text,
                'resources_needed': list(resources_needed),
                'active_incidents': keywords.get('active_incidents', [])
            })

        # --- Frame scaling for UI ---
        with FRAME_LOCK:
            max_w = 800
            if frame.shape[1] > max_w:
                scale = max_w / frame.shape[1]
                frame_small = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
            else:
                frame_small = frame
            CURRENT_FRAME = frame_small.copy()

        time.sleep(inference_delay)
