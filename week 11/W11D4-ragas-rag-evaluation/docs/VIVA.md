# Viva preparation

1. Faithfulness measures whether the answer is supported by retrieved context; Ragas uses an LLM-based assessment to judge claim support.
2. Context precision rewards relevant retrieved chunks and penalizes irrelevant ones; context recall measures whether the retrieved context contains the information needed by the reference answer.
3. Improve answer relevancy by tightening retrieval, tuning chunk size and top-k, improving query rewriting, and using a clearer answer prompt.
