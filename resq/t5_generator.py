from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load T5 model (T5-small or a larger one if you want)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def generate_report_from_incident(incident_data):
    """
    Generate a textual incident report from incident data using T5.
    """
    objects_str = ", ".join(incident_data.get('objects_detected', []))
    multi_incidents = incident_data.get('multi_incident_string', '')
    incident_type = incident_data.get('incident_type', 'Unknown')

    prompt = f"Generate a detailed incident report for a {incident_type}. " \
             f"Detected objects: {objects_str}. Other incidents: {multi_incidents}."

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    report_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return report_text

def generate_next_events(current_events, max_new_events=3):
    """
    Generate the next sequential events given current events using T5.
    
    Args:
        current_events (str): Current sequential events as a string, e.g.
                              "1. Crash (P1); 2. Jam (P3)"
        max_new_events (int): Number of new events to predict

    Returns:
        next_events_list (list of str): List of predicted sequential events
    """
    prompt = f"Given the following incident events: {current_events}. " \
             f"Predict the next {max_new_events} possible sequential events in order."

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=100,
            num_beams=4,
            early_stopping=True,
            do_sample=True,
            top_p=0.9,
            top_k=50
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Split generated text by semicolon or newline to get individual events
    next_events = [e.strip() for e in generated_text.replace("\n", ";").split(";") if e.strip()]
    return next_events[:max_new_events]

# --- Example usage ---
if __name__ == "__main__":
    current_seq_events = "1. Crash (P1); 2. Jam (P3)"
    print("Current events:", current_seq_events)

    next_events = generate_next_events(current_seq_events)
    print("Predicted next events:")
    for i, e in enumerate(next_events, start=3):
        print(f"{i}. {e}")
