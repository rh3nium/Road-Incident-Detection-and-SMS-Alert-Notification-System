import numpy as np

VEHICLE_CLASSES = ['car', 'bicycle', 'truck', 'bus', 'motorcycle']
OBSTACLE_CLASSES = ['pottedplant', 'chair', 'diningtable', 'sofa', 'tvmonitor']
CLUSTER_MAX_DISTANCE = 0.15
JAM_CLUSTER_SIZE = 4

def boxes_touch_or_overlap(box1, box2):
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2
    return not (x2a < x1b or x2b < x1a or y2a < y1b or y2b < y1a)

def boxes_overlap_significantly(box1, box2, threshold=0.02):
    x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    intersection_area = x_overlap * y_overlap
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = area1 + area2 - intersection_area
    if union_area <= 0:
        return False
    return (intersection_area / union_area) > threshold

def cluster_boxes(vehicle_boxes):
    if not vehicle_boxes:
        return 0
    centers = []
    for v in vehicle_boxes:
        x_center = (v['box'][0] + v['box'][2]) / 2
        y_center = (v['box'][1] + v['box'][3]) / 2
        centers.append([x_center, y_center])
    centers = np.array(centers)
    n = len(centers)
    labels = np.arange(n)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < CLUSTER_MAX_DISTANCE:
                labels[labels == labels[j]] = labels[i]
    unique_labels, counts = np.unique(labels, return_counts=True)
    return int(np.max(counts)) if counts.size > 0 else 0  # CRITICAL FIX

def classify_incident(detected_objects, vehicle_boxes, obstacle_boxes, fire_boxes, active_incidents=None):
    if active_incidents is None:
        active_incidents = []

    new_incidents = []

    # Fire
    if len(fire_boxes) > 0:
        new_incidents.append({"type": "Fire", "damage": "Extreme", "priority": 1})

    # Person hit
    person_boxes = [v['box'] for v in vehicle_boxes if v['class'] == 'person']
    vehicle_only = [v for v in vehicle_boxes if v['class'] in VEHICLE_CLASSES]

    for p_box in person_boxes:
        for v in vehicle_only:
            if boxes_overlap_significantly(p_box, v['box'], 0.05):
                new_incidents.append({"type": "Person Hit", "damage": "Extreme", "priority": 1})
                break

    # Crash
    crash_detected = False
    if len(vehicle_only) >= 2:
        for i in range(len(vehicle_only)):
            for j in range(i+1, len(vehicle_only)):
                if boxes_touch_or_overlap(vehicle_only[i]['box'], vehicle_only[j]['box']):
                    crash_detected = True
                    break
            if crash_detected:
                break

    if not crash_detected:
        for v in vehicle_only:
            for o_box in obstacle_boxes:
                if boxes_touch_or_overlap(v['box'], o_box):
                    crash_detected = True
                    break
            if crash_detected:
                break

    if crash_detected:
        new_incidents.append({"type": "Crash", "damage": "High", "priority": 1})

    # Jam
    largest_cluster = cluster_boxes(vehicle_boxes)
    if largest_cluster >= JAM_CLUSTER_SIZE:
        new_incidents.append({"type": "Jam", "damage": "Low", "priority": 3})

    # Normal flow
    if not new_incidents and not active_incidents:
        new_incidents.append({"type": "Normal Flow", "damage": "None", "priority": 4})

    combined_types = {inc['type'] for inc in active_incidents}
    for inc in new_incidents:
        if inc['type'] not in combined_types:
            active_incidents.append(inc)

    active_incidents.sort(key=lambda x: x['priority'])

    multi_incident_str = "; ".join(
        [f"{i+1}. {inc['type']} (P{inc['priority']})" for i, inc in enumerate(active_incidents)]
    )

    return {
        "incident_type": multi_incident_str if active_incidents else "Normal Flow",
        "objects_detected": detected_objects,
        "multi_incident_string": multi_incident_str,
        "active_incidents": active_incidents
    }

