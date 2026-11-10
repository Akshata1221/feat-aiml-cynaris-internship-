QUESTIONS = [
 "How long do I have to request a refund?",
 "What proof is needed for a refund?",
 "When is customer support available?",
 "How do I reset my password?",
 "Where is the password reset link?",
 "When should I report a damaged delivery?",
 "What should I do if my item arrives damaged?",
 "Can I request a refund after 30 days?",
 "What days does support operate?",
 "What information should I provide for a refund?",
]
GROUND_TRUTHS = [
 "Refunds are available within 30 days.",
 "A receipt is required for a refund.",
 "Support is available Monday to Friday, 9 AM to 5 PM.",
 "Use the Forgot Password link on the sign-in page.",
 "The Forgot Password link is on the sign-in page.",
 "Report damaged items within seven days of delivery.",
 "Report the damaged item within seven days of delivery.",
 "The policy states refunds are available within 30 days.",
 "Support operates Monday to Friday.",
 "Provide a receipt for the refund request.",
]

def make_samples(retriever):
    samples = []
    for question, truth in zip(QUESTIONS, GROUND_TRUTHS):
        result = retriever(question)
        samples.append({
            "user_input": question,
            "retrieved_contexts": result["contexts"],
            "response": result["answer"],
            "reference": truth,
        })
    return samples
