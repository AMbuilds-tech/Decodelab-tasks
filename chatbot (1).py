"""
Project 1: Rule-Based AI Chatbot
DecodeLabs - Industrial Training Kit (Batch 2026)

A rule-based chatbot built with pure control flow and a dictionary-based
knowledge base (the "Logic Skeleton" from the training deck), covering all
five spec requirements:
  1. INPUT LOOP     - continuous while cycle
  2. SANITIZATION   - handles case & whitespace
  3. KNOWLEDGE BASE - dictionary with 5+ intents
  4. FALLBACK       - default response for unknown input
  5. EXIT STRATEGY  - clean break command
"""


def sanitize_input(raw_input: str) -> str:
    """Normalize user input: lowercase and strip whitespace."""
    return raw_input.lower().strip()


# Knowledge base: dictionary maps a normalized intent -> response.
# A dictionary gives O(1) lookup instead of a long, unstable if-elif ladder.
responses = {
    "hello": "Hi there! How can I help you today?",
    "hi": "Hello! What can I do for you?",
    "how are you": "I'm just a bunch of if-else statements, but I'm doing great!",
    "what is your name": "I'm ChatBot-01, DecodeLabs' Project 1 chatbot.",
    "what can you do": "I can chat with you using predefined rules. Try me!",
    "help": "Say hello, ask my name, ask how I am, or type 'bye' to exit.",
    "thank you": "You're welcome!",
    "thanks": "Anytime!",
}

EXIT_COMMANDS = {"bye", "exit", "quit"}
FALLBACK_RESPONSE = "I do not understand. Type 'help' to see what I can do."


def get_response(user_input: str) -> str:
    """Look up a response for the sanitized input, falling back if unmatched."""
    return responses.get(user_input, FALLBACK_RESPONSE)


def run_chatbot():
    print("ChatBot-01: Hello! I'm your rule-based assistant. Type 'bye' to exit.")
    while True:
        raw = input("You: ")
        clean = sanitize_input(raw)

        if clean in EXIT_COMMANDS:
            print("ChatBot-01: Goodbye! Have a great day.")
            break

        reply = get_response(clean)
        print(f"ChatBot-01: {reply}")


if __name__ == "__main__":
    run_chatbot()
