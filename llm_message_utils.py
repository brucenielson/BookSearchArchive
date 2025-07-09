# noinspection PyPackageRequirements
from google.api_core.exceptions import ResourceExhausted
# noinspection PyPackageRequirements
from google.generativeai import ChatSession
# noinspection PyPackageRequirements
from google.generativeai.types.generation_types import GenerationConfig, GenerateContentResponse
# noinspection PyPackageRequirements
import google.generativeai as genai
# noinspection PyPackageRequirements
from google.generativeai.types import Tool
from typing import Any, List, Union
import time


def send_message(model: Union[ChatSession, genai.GenerativeModel],
                 message: str,
                 tools: List[Tool] = None,
                 stream: bool = False,
                 config: GenerationConfig = None,
                 **generation_kwargs: Any) -> GenerateContentResponse:

    if config is None and not generation_kwargs:
        # Set up the config with any provided generation parameters
        config: GenerationConfig = GenerationConfig(**generation_kwargs)

    try:
        if isinstance(model, ChatSession):
            # If the model is a ChatSession, use the send_message method
            return model.send_message(message,
                                      generation_config=config,
                                      tools=tools,
                                      stream=stream)
        else:
            # Otherwise, generate content directly without a chat session
            response = model.generate_content(
                contents=message,
                generation_config=config,
                tools=tools,
                stream=stream
            )
            return getattr(response, "text", None) or "[No response text]"

    except ResourceExhausted as e:
        # Handle rate limit errors by pausing and retrying
        print()
        print(f"Rate limit exceeded. Retrying in 15 seconds...")
        time.sleep(15)
        return send_message(model, message, tools=tools, stream=stream, config=config, **generation_kwargs)
    except Exception as e:
        print(f"Error during chat message sending: {e}")
        raise
