import json
import os
from pathlib import Path
from engine import InferenceEngine
from transformers import AutoTokenizer
import sys


def get_config_path():
    """Get the path to the config file in home directory"""
    home_dir = Path.home()
    return home_dir / ".yaie_config.json"


def save_model_to_config(model_name: str):
    """Save the last used model to config"""
    config_path = get_config_path()
    config = {}

    if config_path.exists():
        try:
            with config_path.open('r') as f:
                config = json.load(f)
        except:
            config = {}

    config['last_used_model'] = model_name

    with config_path.open('w') as f:
        json.dump(config, f, indent=2)


def load_last_used_model():
    """Load the last used model from config"""
    config_path = get_config_path()

    if config_path.exists():
        try:
            with config_path.open('r') as f:
                config = json.load(f)
                return config.get('last_used_model')
        except:
            pass

    return None


def run_chat(model_name: str):
    """
    Run an interactive chat session with the model

    Args:
        model_name: Name of the model to load
    """
    print(f"Starting chat with model: {model_name}")
    print("Enter 'quit' to exit the chat.\n")

    # Save the model name to config for future use
    save_model_to_config(model_name)

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
