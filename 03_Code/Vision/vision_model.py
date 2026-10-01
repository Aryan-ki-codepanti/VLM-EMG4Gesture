
import json
import time

from google import genai

from .vision_config import (
    VISION_MODEL,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
)

from .vision_schema import VISION_SCHEMA


def create_vision_client(api_key):
    """
    Create Gemini client for vision inference.
    """
    if not api_key:
        raise ValueError("Gemini API key is required.")

    return genai.Client(api_key=api_key)


def run_vision_inference(client, image, prompt):
    """
    Run vision inference with structured JSON output.

    Retries temporary API failures such as HTTP 503.
    """

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.models.generate_content(
                model=VISION_MODEL,
                contents=[
                    image,
                    prompt,
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": VISION_SCHEMA,
                },
            )

            return json.loads(response.text)

        except Exception as e:

            last_error = e

            print(
                f"Vision inference attempt "
                f"{attempt}/{MAX_RETRIES} failed: {e}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"Vision inference failed after "
        f"{MAX_RETRIES} attempts."
    ) from last_error
