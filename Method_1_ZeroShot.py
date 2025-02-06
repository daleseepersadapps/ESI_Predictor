import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the Qwen2-1.5b model and tokenizer
model_name = "Qwen/Qwen2-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Define the prompt template for zero-shot ESI prediction
def generate_esi_prediction(triage_text):
    prompt = f"""
    Given the following emergency department triage narrative, assign an Emergency Severity Index (ESI) level (1-5) and provide a reasoning explanation:
    
    Triage Narrative:
    "{triage_text}"
    
    Response Format:
    - ESI Level: <1/2/3/4/5>
    - Reasoning: <Explain the reasoning behind the classification>
    
    Provide a step-by-step reasoning based on patient acuity, vital signs, and resource needs.
    """

    # Generate response
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_length=512, temperature=0.7, top_p=0.9)
    
    # Decode and return the model output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example triage narratives
triage_cases = [
    "Patient is a 65-year-old male presenting with crushing substernal chest pain radiating to the left arm. He appears diaphoretic and short of breath. BP: 90/60 mmHg, HR: 110 bpm, RR: 24/min, SpO2: 92%. Pain score: 9/10.",
    "Patient is a 25-year-old female with a minor laceration on the right forearm. No active bleeding. Vitals are stable, and she reports mild discomfort. No underlying conditions.",
]

# Run zero-shot predictions
for case in triage_cases:
    result = generate_esi_prediction(case)
    print("\n---- Generated Prediction ----")
    print(result)
