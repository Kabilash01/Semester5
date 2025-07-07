# C:\E0323036-S5\chatbot\app.py

import google.generativeai as genai
import os

# --- IMPORTANT ---
# This is the line that fixes your error.
#
# TO GET AN API KEY:
# 1. Go to Google AI Studio: https://aistudio.google.com/app/apikey
# 2. Click "Create API key" and copy the key.
# 3. Paste your key directly into the quotes below.
#
# For better security, you can set this as an environment variable instead.
# The code below will first try to get the key from an environment variable
# and if it fails, it will look for it directly in the string.
try:
    # Recommended: Set GOOGLE_API_KEY environment variable
    api_key = os.environ["GOOGLE_API_KEY"]
except KeyError:
    # Fallback: Paste your API key directly here if not using env var
    api_key = "AIzaSyAnMn3nw9D4yC6i0h1GClnQykxKI3mE2Lg"

# Ensure you have a key before continuing
if api_key == "AIzaSyAgj1LhnMOrMlqAoS__KB1hltYmVzxA49g":
    print("ERROR: API key not found.")
 
    exit() # Exit the script if the key is not set

# Configure the library with your API key
genai.configure(api_key=api_key)


def run_chatbot():
    """
    Initializes and runs the interactive chatbot session.
    """
    print("--- Starting Gemini Chatbot ---")
    print("Initializing model...")

    # Set up the model
    generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    # Initialize the Generative Model
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro
            generation_config=generation_config,
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Please ensure your API key is valid and has permissions.")
        return

    # Start a chat session
    chat = model.start_chat(history=[])

    print("Model initialized. You can now start chatting.")
    print("Type 'quit' or 'exit' to end the session.\n")

    while True:
        # Get user input
        user_prompt = input("You: ")

        # Check for exit command
        if user_prompt.lower() in ["quit", "exit"]:
            print("\n--- Chat session ended. Goodbye! ---")
            break

        # Send the message to the model and get the response
        try:
            response = chat.send_message(user_prompt, stream=True)
            
            print("Gemini: ", end="")
            # Print the streamed response
            for chunk in response:
                print(chunk.text, end="")
            print("\n") # Add a newline after the full response

        except Exception as e:
            print(f"An error occurred: {e}")
            print("The chat session might have been terminated. Please restart.")
            break


# --- Main execution block ---
if __name__ == "__main__":
    run_chatbot()