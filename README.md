# Enhancing Emergency Triage: A Novel Fine-Tuned LLM for Accurate and Interpretable ESI Prediction

D. Seepersad, MD

Traditional machine learning methods for Emergency Severity Index (ESI) prediction often lack interpretability, relying on complex statistical correlations or feature importance rankings that are not easily reviewed in a clinical setting [1]. To safely integrate machine-made medical decisions into practice, human-readable reasoning, such as clearly articulated steps or explanations for predictions, is essential for clinical trust and verification. Flexible and interpretable models are needed. This study compares large language models (LLMs) to nurse-assigned ESI levels and introduces a novel fine-tuned model integrating reasoning outputs and contextual learning to enhance accuracy and interpretability.

We utilized a dataset from US Acute Care Solutions (USACS), comprising clinical data from emergency department visits across multiple hospitals. A validated reference standard for ESI levels was established using resource utilization, patient disposition, and mortality, aligned with ESI Guidelines [2]. Nurse-assigned ESI levels were compared to the reference standard for baseline performance. A zero-shot LLM approach was then applied to predict ESI levels from unstructured triage narratives. Finally, a fine-tuned LLM model was developed, leveraging reasoning outputs to improve prediction accuracy and interpretability. Methods were evaluated based on accuracy, interpretability, and clinical relevance under real-world conditions.

Preliminary findings show the fine-tuned LLM model outperforms both the zero-shot approach and nurse performance in ESI prediction accuracy while providing interpretable reasoning. Benchmarking LLMs against clinical practice under realistic conditions highlights their potential as decision-support tools to improve triage efficiency and consistency in emergency department workflows.

