from .vision_prompt import VISION_PROMPT
from .vision_model import run_vision_inference
from .vision_parser import parse_vision_output


def infer_vision(client, image):
    """
    Run complete Vision inference pipeline.

    image
        PIL image

    Returns
    -------
    dict
        Normalized structured vision output.
    """

    raw_output = run_vision_inference(
        client=client,
        image=image,
        prompt=VISION_PROMPT
    )

    parsed_output = parse_vision_output(raw_output)

    return parsed_output
