import re
from typing import Dict
import openai  # Ensure you have installed the openai package and set your API key

class ESITriageReward:
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize with optional weights for each sub-reward.
        Default weights are set to 1.0 for all reward types.
        """
        self.weights = {
            "accuracy": 1.0,
            "alignment": 1.0,
            "safety": 1.0,
            "explainability": 1.0,
            "bias": 1.0,
            "uncertainty": 1.0
        }
        if weights:
            self.weights.update(weights)

    def reward_accuracy(self, true_esi: int, predicted_esi: int) -> float:
        """
        Accuracy Reward:
          - +10 for an exact match.
          - If incorrect, apply a penalty proportional to the difference.
          - Extra penalty (×2) if under–triaging (i.e., predicting a less urgent score).
        """
        if predicted_esi == true_esi:
            return 10.0

        diff = abs(predicted_esi - true_esi)
        # Under–triage: predicted ESI > true ESI
        if predicted_esi > true_esi:
            penalty = 2 * diff
        else:
            penalty = diff
        return -penalty

    def reward_reasoning_alignment(self, reasoning: str) -> float:
        """
        Reasoning Alignment Reward:
          - Uses an LLM to assess if the explanation aligns with established ESI guidelines.
          - The prompt instructs the LLM to evaluate the reasoning in terms of clarity and
            the inclusion of key ESI factors (e.g., vital signs, chief complaint, resource utilization).
          - Expected score is on a scale (for example, -5 to +5).
        """
        score = self.call_llm_assessment_alignment(reasoning)
        return score

    def reward_safety(self, true_esi: int, predicted_esi: int, reasoning: str) -> float:
        """
        Safety and Conservatism Reward:
          - Uses an LLM to assess the safety of the triage decision.
          - The prompt instructs the LLM to consider the difference between the true and predicted ESI,
            and whether the reasoning errs on the side of caution.
          - Expected score is on a scale (e.g., -2 to +2).
        """
        score = self.call_llm_assessment_safety(true_esi, predicted_esi, reasoning)
        return score

    def reward_explainability(self, reasoning: str) -> float:
        """
        Transparency and Explainability Reward:
          - Calls an LLM to assess the quality of the explanation text.
          - The LLM is prompted to evaluate clarity, conciseness, and logical structure, and
            return only a numerical score (e.g., -2 to +2).
        """
        score = self.call_llm_assessment_explainability(reasoning)
        return score

    def reward_bias_mitigation(self, reasoning: str) -> float:
        """
        Bias Mitigation Reward:
          - Uses an LLM to assess the explanation for the presence of irrelevant or potentially biased
            references to sensitive demographic information.
          - The prompt instructs the LLM to return a numerical score indicating if the explanation is bias–free
            (e.g., -3 to 0, where negative indicates bias).
        """
        score = self.call_llm_assessment_bias(reasoning)
        return score

    def reward_uncertainty_handling(self, reasoning: str, input_is_ambiguous: bool = False) -> float:
        """
        Handling Uncertainty Reward:
          - When the input is ambiguous, reward if uncertainty-related terms are present.
          - Penalize if the explanation is overconfident despite ambiguous inputs.
        """
        uncertainty_keywords = ["uncertain", "possibly", "might", "unsure", "ambiguous", "low confidence"]
        if input_is_ambiguous:
            return 2.0 if any(kw in reasoning.lower() for kw in uncertainty_keywords) else -2.0
        return 0.0

    def call_llm_assessment_explainability(self, explanation: str) -> float:
        """
        Call an LLM to assess explanation quality based on clarity, conciseness, and structure.
        Returns a numerical score on a predefined scale.
        """
        prompt = (
            "You are an expert medical evaluator. Please assess the following explanation for clarity, conciseness, "
            "and logical structure on a scale from -2 (poor) to +2 (excellent). Return only the numerical score.\n\n"
            f"Explanation: {explanation}"
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an assistant evaluating medical explanation texts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0  # Deterministic output
            )
            score_str = response["choices"][0]["message"]["content"].strip()
            score = float(score_str)
        except Exception as e:
            print("Error calling LLM for explainability assessment:", e)
            score = 0.0
        return score

    def call_llm_assessment_alignment(self, reasoning: str) -> float:
        """
        Call an LLM to assess whether the reasoning aligns with ESI guidelines.
        Expected to evaluate the inclusion of key factors (vital signs, chief complaint, resource utilization, etc.)
        and return a numerical score (e.g., -5 to +5).
        """
        prompt = (
            "You are an expert medical evaluator. Please assess the following reasoning for alignment with "
            "established Emergency Severity Index (ESI) guidelines. Consider whether the explanation includes "
            "relevant factors such as vital signs, chief complaint, and resource utilization. Rate the alignment "
            "on a scale from -5 (poor alignment) to +5 (excellent alignment). Return only the numerical score.\n\n"
            f"Reasoning: {reasoning}"
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an assistant evaluating ESI reasoning alignment."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            score_str = response["choices"][0]["message"]["content"].strip()
            score = float(score_str)
        except Exception as e:
            print("Error calling LLM for reasoning alignment assessment:", e)
            score = 0.0
        return score

    def call_llm_assessment_safety(self, true_esi: int, predicted_esi: int, reasoning: str) -> float:
        """
        Call an LLM to assess the safety of the triage decision.
        The prompt includes the true and predicted ESI scores as well as the reasoning,
        asking the LLM to evaluate whether the decision errs on the side of caution.
        Returns a numerical score (e.g., -2 to +2).
        """
        prompt = (
            "You are an expert medical evaluator. Evaluate the safety of the following triage decision. "
            "The true Emergency Severity Index (ESI) is provided along with the predicted ESI and the reasoning behind the prediction. "
            "A safe decision should err on the side of caution (i.e., not under–triaging). "
            "Rate the safety of the decision on a scale from -2 (unsafe, under–triage) to +2 (safe, conservative decision). "
            "Return only the numerical score.\n\n"
            f"True ESI: {true_esi}\nPredicted ESI: {predicted_esi}\nReasoning: {reasoning}"
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an assistant evaluating triage safety."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            score_str = response["choices"][0]["message"]["content"].strip()
            score = float(score_str)
        except Exception as e:
            print("Error calling LLM for safety assessment:", e)
            score = 0.0
        return score

    def call_llm_assessment_bias(self, reasoning: str) -> float:
        """
        Call an LLM to assess the explanation for potential biases.
        The LLM is asked to determine if the explanation unnecessarily references sensitive demographic information,
        and to return a score on a scale (e.g., -3 for biased to 0 for bias-free).
        """
        prompt = (
            "You are an expert in medical ethics. Please assess the following explanation for any unnecessary or biased "
            "references to sensitive demographic information (e.g., race, ethnicity, gender, age, etc.). "
            "Rate the explanation on a scale from -3 (highly biased) to 0 (bias-free). Return only the numerical score.\n\n"
            f"Explanation: {reasoning}"
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an assistant evaluating bias in medical explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            score_str = response["choices"][0]["message"]["content"].strip()
            score = float(score_str)
        except Exception as e:
            print("Error calling LLM for bias assessment:", e)
            score = 0.0
        return score

    def compute_total_reward(self, true_esi: int, predicted_esi: int,
                             reasoning: str, input_is_ambiguous: bool = False) -> float:
        """
        Compute the total reward as the weighted sum of all sub-rewards.
        """
        r_acc         = self.weights["accuracy"] * self.reward_accuracy(true_esi, predicted_esi)
        r_alignment   = self.weights["alignment"] * self.reward_reasoning_alignment(reasoning)
        r_safety      = self.weights["safety"] * self.reward_safety(true_esi, predicted_esi, reasoning)
        r_explain     = self.weights["explainability"] * self.reward_explainability(reasoning)
        r_bias        = self.weights["bias"] * self.reward_bias_mitigation(reasoning)
        r_uncertainty = self.weights["uncertainty"] * self.reward_uncertainty_handling(reasoning, input_is_ambiguous)

        total_reward = r_acc + r_alignment + r_safety + r_explain + r_bias + r_uncertainty
        return total_reward

# ===== Example usage =====
if __name__ == "__main__":
    # Configure reward model (optionally adjust weights here)
    reward_model = ESITriageReward()

    # Example inputs:
    true_esi = 2
    predicted_esi = 3  # This represents an under–triage scenario.
    reasoning = (
        "The patient presents with abnormal vital signs and a severe chief complaint. "
        "Based on ESI guidelines and resource utilization estimates, the case appears urgent. "
        "There is some uncertainty regarding the extent of the underlying condition."
    )
    input_is_ambiguous = True

    total = reward_model.compute_total_reward(true_esi, predicted_esi, reasoning, input_is_ambiguous)
    print("Total reward:", total)
