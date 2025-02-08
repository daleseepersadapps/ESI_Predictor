import re

def reward_accuracy(true_esi, predicted_esi):
    """
    Accuracy Reward:
      - +10 for an exact match.
      - If incorrect, apply a penalty proportional to the difference.
      - Extra penalty if the model under–triages (i.e. predicts a higher ESI, meaning a less urgent assessment,
        than the true ESI).
    """
    if predicted_esi == true_esi:
        return 10
    else:
        diff = abs(predicted_esi - true_esi)
        # Under–triage: predicted ESI > true ESI (i.e. patient is more acute than predicted)
        if predicted_esi > true_esi:
            penalty = 2 * diff   # higher penalty for under–triage
        else:
            penalty = diff       # less penalty for over–triage (err on the side of caution)
        return -penalty

def reward_reasoning_alignment(reasoning):
    """
    Reasoning Alignment Reward:
      - Check that the reasoning mentions key ESI factors.
      - In this example, we look for keywords such as 'vital signs', 'chief complaint',
        'resource utilization', 'ESI guidelines', and 'triage'.
      - Reward if at least a threshold number of keywords are present; otherwise, apply a penalty.
    """
    required_keywords = [
        "vital signs", 
        "chief complaint", 
        "resource utilization", 
        "esi guidelines", 
        "triage"
    ]
    matches = sum(1 for kw in required_keywords if kw in reasoning.lower())
    
    # For example, if 3 or more keywords are found, we reward; else, we penalize.
    return 5 if matches >= 3 else -5

def reward_safety(true_esi, predicted_esi):
    """
    Safety and Conservatism Reward:
      - It is safer to over–triage (i.e. predict a lower ESI than the true value, implying more urgency)
        than to under–triage.
      - Here we add a small bonus if the prediction is conservative (predicted ESI <= true ESI).
      - Otherwise, a penalty is applied.
    """
    if predicted_esi <= true_esi:
        return 2   # bonus for erring on the side of caution (over–triage)
    else:
        return -2  # penalty for under–triage

def reward_explainability(reasoning):
    """
    Transparency and Explainability Reward:
      - Reward if the reasoning explanation is clear, concise, and structured.
      - As a simple heuristic, we check the number of sentences (by splitting on periods).
      - Here, we reward if the explanation has between 2 and 5 sentences; otherwise, penalize.
    """
    # Split on period and remove empty strings after stripping whitespace.
    sentences = [s.strip() for s in reasoning.split('.') if s.strip()]
    num_sentences = len(sentences)
    
    return 2 if 2 <= num_sentences <= 5 else -2

def reward_bias_mitigation(reasoning):
    """
    Bias Mitigation Reward:
      - Penalize responses that reference sensitive demographic information if it is not needed for triage.
      - This is a simple keyword–based check. You may wish to improve this check or use more advanced methods.
    """
    sensitive_terms = ["race", "ethnicity", "gender", "age", "sex", "socioeconomic", "income"]
    # Check if any sensitive term is mentioned in the reasoning.
    if any(term in reasoning.lower() for term in sensitive_terms):
        return -3
    else:
        return 0

def reward_uncertainty_handling(reasoning, input_is_ambiguous=False):
    """
    Handling Uncertainty Reward:
      - When the input information is ambiguous, the model should express uncertainty.
      - If the input is flagged as ambiguous, check for uncertainty–related terms in the reasoning.
      - Reward if such terms are present; otherwise, apply a penalty.
    """
    uncertainty_keywords = ["uncertain", "possibly", "might", "unsure", "ambiguous", "low confidence"]
    
    if input_is_ambiguous:
        if any(kw in reasoning.lower() for kw in uncertainty_keywords):
            return 2
        else:
            return -2
    else:
        return 0

def compute_total_reward(true_esi, predicted_esi, reasoning, input_is_ambiguous=False):
    """
    Compute the total reward as the sum of all sub–rewards.
    You can adjust the weights or even combine some rewards multiplicatively as needed.
    """
    r_acc         = reward_accuracy(true_esi, predicted_esi)
    r_alignment   = reward_reasoning_alignment(reasoning)
    r_safety      = reward_safety(true_esi, predicted_esi)
    r_explain     = reward_explainability(reasoning)
    r_bias        = reward_bias_mitigation(reasoning)
    r_uncertainty = reward_uncertainty_handling(reasoning, input_is_ambiguous)
    
    total_reward = (r_acc + r_alignment + r_safety + r_explain + r_bias + r_uncertainty)
    return total_reward

# ===== Example usage =====
if __name__ == "__main__":
    # Example inputs:
    true_esi = 2
    predicted_esi = 3  # For example, this is an under–triage if the patient is truly high–acuity.
    
    reasoning = (
        "The patient presents with abnormal vital signs and a severe chief complaint. "
        "Based on ESI guidelines and resource utilization estimates, the case appears urgent. "
        "There is some uncertainty regarding the extent of the underlying condition."
    )
    
    # Assume the input description was ambiguous.
    input_is_ambiguous = True
    
    total = compute_total_reward(true_esi, predicted_esi, reasoning, input_is_ambiguous)
    print("Total reward:", total)
