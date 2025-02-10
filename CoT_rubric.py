import re

def score_chain_of_thought(reasoning: str) -> (int, dict):
    """
    Evaluate the chain-of-thought reasoning based on a rubric with 7 criteria.
    Each criterion is scored from 0 to 2, for a maximum total score of 14.
    Returns the total score and a dictionary with scores for each criterion.
    """
    total_score = 0
    details = {}

    # 1. Guideline Alignment
    guideline_keywords = ["vital signs", "chief complaint", "resource", "ESI"]
    count_guidelines = sum(1 for kw in guideline_keywords if kw in reasoning.lower())
    if count_guidelines >= 3:
        score = 2
    elif count_guidelines >= 1:
        score = 1
    else:
        score = 0
    total_score += score
    details["Guideline Alignment"] = score

    # 2. Logical Flow
    transitional_keywords = ["because", "therefore", "thus", "hence", "as a result"]
    count_transitions = sum(1 for kw in transitional_keywords if kw in reasoning.lower())
    if count_transitions >= 2:
        score = 2
    elif count_transitions == 1:
        score = 1
    else:
        score = 0
    total_score += score
    details["Logical Flow"] = score

    # 3. Clarity and Conciseness
    sentences = [s.strip() for s in reasoning.split('.') if s.strip()]
    num_sentences = len(sentences)
    # Ideally, a clear explanation might have between 2 and 5 sentences.
    if 2 <= num_sentences <= 5:
        score = 2
    elif num_sentences < 2:
        score = 0  # too brief
    else:
        score = 1  # too verbose
    total_score += score
    details["Clarity and Conciseness"] = score

    # 4. Evidence-Based Justification
    evidence_keywords = ["abnormal", "elevated", "low", "high", "pain", "symptom", "fever"]
    count_evidence = sum(1 for kw in evidence_keywords if kw in reasoning.lower())
    if count_evidence >= 2:
        score = 2
    elif count_evidence == 1:
        score = 1
    else:
        score = 0
    total_score += score
    details["Evidence-Based Justification"] = score

    # 5. Uncertainty Handling
    uncertainty_keywords = ["uncertain", "possibly", "might", "unsure", "ambiguous", "unclear"]
    if any(kw in reasoning.lower() for kw in uncertainty_keywords):
        score = 2
    else:
        score = 0
    total_score += score
    details["Uncertainty Handling"] = score

    # 6. Safety Focus
    safety_keywords = ["safety", "caution", "risk", "urgent", "critical", "over-triage"]
    if any(kw in reasoning.lower() for kw in safety_keywords):
        score = 2
    else:
        score = 0
    total_score += score
    details["Safety Focus"] = score

    # 7. Absence of Bias
    bias_keywords = ["race", "ethnicity", "gender", "age", "socioeconomic", "income"]
    if any(kw in reasoning.lower() for kw in bias_keywords):
        score = 0
    else:
        score = 2
    total_score += score
    details["Absence of Bias"] = score

    return total_score, details

# === Example Usage ===
if __name__ == "__main__":
    sample_reasoning = (
        "The patient presents with abnormal vital signs and a severe chief complaint, "
        "which according to ESI guidelines indicates the need for urgent care. "
        "Because the vital signs are elevated and the patient exhibits significant symptoms, "
        "immediate resource allocation is justified. "
        "There is some uncertainty in the exact level of care needed, so a cautious approach is taken to ensure safety."
    )
    
    total, breakdown = score_chain_of_thought(sample_reasoning)
    print("Chain-of-Thought Score:", total)
    for criterion, score in breakdown.items():
        print(f"  {criterion:30s}: {score}")
