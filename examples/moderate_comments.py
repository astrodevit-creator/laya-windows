"""Triage social-media comments locally: language-agnostic, no API calls."""

import json

from laya_windows import load

agent = load()
questions = {
    "intent": {
        "type": "choice",
        "instructions": "What does this comment want?",
        "criteria": {
            "purchase": "wants to buy, asks price or delivery",
            "complaint": "unhappy, scam, refund, bad quality",
            "question": "asks about the product",
            "spam": "unrelated promotion or links",
        },
    },
}
comments = [
    "Combien le prix avec livraison à Casablanca ?",
    "This is a scam, I never received my order!!",
    "Does it work on dry skin?",
    "Follow me for free crypto signals 🚀",
]
for text in comments:
    a = agent.predict(text, questions)["answers"]
    print(json.dumps({"comment": text, "intent": a["intent"]["choice"],
                      "confidence": a["intent"]["confidence"]}, ensure_ascii=False))
