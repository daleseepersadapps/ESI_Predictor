import re
import os
import requests
from typing import Dict

class ESITriageReward:
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize the reward model with optional weights for each sub-reward.
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
        
        # Set the DeepSeek-R1 API endpoint and API key from environment variables
        self.api_url = os.getenv("DEEPSEEK_R1_API_URL", "https://api.microsoftai-foundry.com/deepseek-r1")
        self.api_key = os.getenv("DEEPSEEK_R1_API_KEY", "YOUR_API_KEY")

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
        if predicted_esi > true_esi:  # under–triage: predicted less urgent than true
            penalty = 2 * diff
        else:
            penalty = diff
        return -penalty

    def reward_reasoning_alignment(self, reasoning: str) -> float:
        """
        Reasoning Alignment Reward:
          - Uses DeepSeek-R1 to assess whether the reasoning aligns with established ESI guidelines.
          - Expected score scale is, for example, from -5 (poor alignment) to +5 (excellent alignment).
        """
        prompt = (
            "You are an expert medical evaluator. Please assess the following reasoning for alignment with "
            "established Emergency Severity Index (ESI) guidelines. Consider whether the explanation includes "
            "relevant factors such as vital signs, chief complaint, and resource utilization. Rate the alignment "
            "on a scale from -5 (poor alignment) to +5 (excellent alignment). Return only the numerical score.\n\n"
            f"Reasoning: {reasoning}"
        )
        return self.call_deepseek_api(prompt)

    def reward_safety(self, true_esi: int, predicted_esi: int, reasoning: str) -> float:
        """
        Safety and Conservatism Reward:
          - Uses DeepSeek-R1 to assess the safety of the triage decision.
          - The LLM considers both the difference between the true and predicted ESI and whether the reasoning errs on the side of caution.
          - Expected score scale is, for example, from -2 (unsafe, under–triage) to +2 (safe, conservative decision).
        """
        prompt = (
            "You are an expert medical evaluator. Evaluate the safety of the following triage decision. "
            "The true Emergency Severity Index (ESI) is provided along with the predicted ESI and the reasoning behind the prediction. "
            "A safe decision should err on the side of caution (i.e., not under–triaging). "
            "Rate the safety of the decision on a scale from -2 (unsafe, under–triage) to +2 (safe, conservative decision). "
            "Return only the numerical score.\n\n"
            f"True ESI: {true_esi}\nPredicted ESI: {predicted_esi}\nReasoning: {reasoning}"
        )
        return self.call_deepseek_api(prompt)

    def reward_explainability(self, reasoning: str) -> float:
        """
        Transparency and Explainability Reward:
          - Uses DeepSeek-R1 to assess the quality of the explanation text.
          - The prompt instructs DeepSeek-R1 to evaluate clarity, conciseness, and logical structure,
            returning a numerical score on a scale from -2 (poor) to +2 (excellent).
        """
        prompt = (
            "You are an expert medical evaluator. Please assess the following explanation for clarity, conciseness, "
            "and logical structure on a scale from -2 (poor) to +2 (excellent). Return only the numerical score.\n\n"
            f"Explanation: {reasoning}"
        )
        return self.call_deepseek_api(prompt)

    def reward_bias_mitigation(self, reasoning: str) -> float:
        """
        Bias Mitigation Reward:
          - Uses DeepSeek-R1 to assess the explanation for any unnecessary or biased references to sensitive demographic information.
          - Expected scale is from -3 (highly biased) to 0 (bias-free).
        """
        prompt = (
            "You are an expert in medical ethics. Please assess the following explanation for any unnecessary or biased "
            "references to sensitive demographic information (e.g., race, ethnicity, gender, age, etc.). "
            "Rate the explanation on a scale from -3 (highly biased) to 0 (bias-free). Return only the numerical score.\n\n"
            f"Explanation: {reasoning}"
        )
        return self.call_deepseek_api(prompt)

    def reward_uncertainty_handling(self, reasoning: str, input_is_ambiguous: bool = False) -> float:
        """
        Handling Uncertainty Reward:
          - When the input is ambiguous, this function checks for the presence of uncertainty-related terms.
          - Rewards the inclusion of such terms (score +2) and penalizes their absence (score -2) when ambiguity is flagged.
        """
        uncertainty_keywords = ["uncertain", "possibly", "might", "unsure", "ambiguous", "low confidence"]
        if input_is_ambiguous:
            return 2.0 if any(kw in reasoning.lower() for kw in uncertainty_keywords) else -2.0
        return 0.0

    def call_deepseek_api(self, prompt: str) -> float:
        """
        Sends a prompt to the DeepSeek-R1 API and returns the numerical score.
        Adjust the request payload and response parsing as needed to match the API specification.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "model": "deepseek-r1",
            "temperature": 0  # For deterministic output
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            # Assuming the API returns a JSON object like: {"score": "1.5"} or {"score": 1.5}
            score_value = result.get("score")
            score = float(score_value)
        except Exception as e:
            print("Error calling DeepSeek-R1 API:", e)
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
    # Optionally set DEEPSEEK_R1_API_URL and DEEPSEEK_R1_API_KEY as environment variables.
    reward_model = ESITriageReward()

    # Example inputs:
    true_esi = 2
    predicted_esi = 3  # Represents an under–triage scenario.
    reasoning = (
        "The patient presents with abnormal vital signs and a severe chief complaint. "
        "Based on ESI guidelines and resource utilization estimates, the case appears urgent. "
        "There is some uncertainty regarding the extent of the underlying condition."
    )
    input_is_ambiguous = True

    total = reward_model.compute_total_reward(true_esi, predicted_esi, reasoning, input_is_ambiguous)
    print("Total reward:", total)
