from engine import InferenceEngine
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

            # Call engine to get response
            response = engine.chat_completion(conversation_history)
            
            # Extract assistant message
            # Structure matches OpenAI format: {'choices': [{'message': {'content': ...}}]}
            try:
                assistant_content = response["choices"][0]["message"]["content"]
                
                # Print response
                print(f"Assistant: {assistant_content}\n")
                
                # Add to history
                conversation_history.append({"role": "assistant", "content": assistant_content})
                
            except (KeyError, IndexError):
                print("Assistant: (Error generating response)\n")

        except KeyboardInterrupt:
            print("\nEnding chat session.")
            break
        except Exception as e:
            print(f"Error during chat: {e}")
            print("Please try again or exit the chat.")
