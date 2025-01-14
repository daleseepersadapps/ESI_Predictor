# ESI_Predictor

The Emergency Severity Index (ESI) is a critical triage tool used in emergency departments (EDs) to prioritize patients based on acuity and resource needs. While prior studies have explored traditional machine learning methods for ESI prediction, their limited applicability in real-world settings underscores the need for more flexible and interpretable models. This study evaluates the potential of large language models (LLMs) to predict ESI levels at the point of triage using only data available at patient arrival, such as demographics, chief complaints, mode of arrival, and vital signs.

Structured electronic medical record (EMR) data is converted into unstructured triage narratives, which are utilized across three distinct approaches: (1) a zero-shot method where the LLM generates a triage level directly from the unstructured text without additional context, (2) a prompt-engineering method where an adapted ESI handbook is integrated to guide the model's reasoning, and (3) fine-tuning, where the LLM is trained to produce both a chain of thought explanation followed by ESI level. Each method is evaluated for accuracy, interpretability, and clinical relevance, with a focus on replicating real-world triage scenarios.

(Hypothetical) Preliminary findings suggest that while zero-shot methods provide a rapid baseline, prompt engineering with the ESI handbook enhances interpretability and alignment with clinical guidelines. Fine-tuning further improves classification accuracy and reasoning quality. By testing multiple techniques under realistic conditions, this study provides valuable insights into the capabilities of LLMs as decision-support tools in ED workflows and their potential to augment triage efficiency and consistency.
