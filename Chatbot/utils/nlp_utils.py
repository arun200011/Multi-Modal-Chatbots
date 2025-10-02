# utils/nlp_utils.py
import json
import random

class FAQChatbot:
    def __init__(self, intents_file="data/intents.json"):
        with open(intents_file, "r") as f:
            self.intents = json.load(f)["intents"]

    def get_response(self, user_input: str) -> str:
        user_input = user_input.lower()

        for intent in self.intents:
            for pattern in intent["patterns"]:
                if pattern.lower() in user_input:
                    return random.choice(intent["responses"])

        return "🤔 Sorry, I didn’t understand. Can you rephrase?"
