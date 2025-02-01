import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig


def get_secret(secret_file: str) -> str:
    """
    Read a secret from a file.

    Args:
        secret_file (str): Path to the file containing the secret.

    Returns:
        str: The content of the secret file, or an empty string if an error occurs.
    """
    try:
        with open(secret_file, 'r') as file:
            secret_text: str = file.read().strip()
    except FileNotFoundError:
        print(f"The file '{secret_file}' does not exist.")
        secret_text = ""
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

    return secret_text


secret = get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
genai.configure(api_key=secret)

# Create the model
generation_config = GenerationConfig(
    temperature=1,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
    response_mime_type="text/plain",
)

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction="You are a Dungeon Master crafting an epic adventure for a party of three adventurers. The campaign should be filled with perilous quests, mythical creatures, and ancient magic. The story should be immersive, full of rich lore, and feature engaging character interactions. The party consists of a brave warrior, a cunning rogue, and a wise mage, each with their own motivations and backstories. The adventure should unfold dynamically, with meaningful choices that impact the world around them.",
)

chat_session = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [
                "Let's begin our adventure.",
            ],
        },
        {
            "role": "model",
            "parts": [
                "Welcome, adventurers, to the realm of Eldoria—a land teeming with magic, mystery, and untold dangers. Your journey begins in the bustling city of Valdris, where the three of you have gathered at the Flickering Lantern, a well-worn tavern known for harboring whispers of lost treasures and unsolved mysteries.\n\n"
                "**Meet the Party:**\n\n"
                "1. **Thorne, the Stalwart Warrior:** A battle-hardened fighter, clad in scarred plate armor, wielding a greatsword that has seen countless battles. Thorne seeks glory and redemption, haunted by a past failure that cost his comrades dearly.\n\n"
                "2. **Lyra, the Shadowed Blade:** A nimble rogue with a sharp wit and an even sharper dagger. Lyra thrives in the shadows, her past shrouded in secrecy. Her motives remain unclear, but her skills in deception and lockpicking are invaluable.\n\n"
                "3. **Eldrin, the Arcane Scholar:** A wise elven mage who wields powerful magic drawn from ancient tomes. Eldrin's thirst for knowledge is insatiable, and he seeks an artifact said to grant insight into the forgotten history of the world.\n\n"
                "**The Call to Adventure:** As you sit around a worn wooden table, a hooded figure approaches. He speaks in a hushed tone, offering a quest that could change the fate of the realm—a missing relic, stolen from the king’s vault, rumored to hold the key to an ancient prophecy.\n\n"
                "What will you do? Accept the quest, probe the stranger for more information, or carve your own path? The adventure is yours to shape.\n\n"
                "**How shall we proceed? The fate of Eldoria awaits your decision.**",
            ],
        },
    ]
)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat...")
        break

    response = chat_session.send_message(user_input)
    print(f"DM: {response.text}\n")
