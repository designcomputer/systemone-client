"""Example usage of the systemone client (all three question types)."""

from systemone import systemone

response = systemone(
    model="tev1:4b",
    state={"ticket": "I was charged twice. Please refund the extra payment. It's urgent!"},
    questions={
        # noul: yes/no -> probability 0-1
        "refund": {
            "type": "noul",
            "instructions": "Is the customer requesting a refund?",
            "criteria": {
                "true": "The customer requests a refund.",
                "false": "No refund is requested.",
            },
        },
        # choice: labeled classification -> label + probabilities + confidence
        "label": {
            "type": "choice",
            "instructions": "Which label fits this ticket?",
            "criteria": {
                "billing": "Payments and refunds",
                "bug": "Software errors",
                "account": "Login and account access",
            },
        },
        # score: rubric -> score on the 0-(N-1) scale
        "urgency": {
            "type": "score",
            "instructions": "How urgently does this ticket need a response?",
            "criteria": [
                "Routine: no time pressure",
                "Soon: a customer is inconvenienced",
                "Immediate: a critical service is unavailable",
            ],
        },
    },
    keep_alive=-1,  # keep the model loaded for subsequent calls
)

print(f"model:   {response.model}")
print(f"refund:  {response.answers['refund'].noul:.4f}")
print(f"label:   {response.answers['label'].choice} "
      f"(confidence {response.answers['label'].confidence:.4f}, "
      f"probs {response.answers['label'].probabilities})")
print(f"urgency: {response.answers['urgency'].score:.4f} (0-2 scale)")
print(f"usage:   {response.usage}")
