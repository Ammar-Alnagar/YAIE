from ..engine import InferenceEngine
from transformers import AutoTokenizer
import sys


def run_chat(model_name: str):
    """
    Run an interactive chat session with the model

    Args:
        model_name: Name of the model to load
    """
    print(f"Starting chat with model: {model_name}")
    print("Enter 'quit' to exit the chat.\n")

    # Initialize the inference engine
    engine = InferenceEngine(model_name)

    # Initialize conversation history
    conversation_history = []

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Ending chat session.")
                break

            # Add user message to conversation
            conversation_history.append({"role": "user", "content": user_input})

            # TODO: Implement the chat response generation logic
            # This should:
            # 1. Format the conversation history properly for the model
            # 2. Call the engine's chat_completion or generate method
            # 3. Handle the response appropriately
            # 4. Add the model's response to conversation history
            # 5. Display the model's response to the user
            # 6. Manage context length to avoid exceeding model limits

            # For now, raise an error to indicate this needs implementation
            print("Assistant: [Response would appear here once implemented]")

        except KeyboardInterrupt:
            print("\nEnding chat session.")
            break
        except Exception as e:
            print(f"Error during chat: {e}")
            print("Please try again or exit the chat.")
