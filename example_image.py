"""Image test for vision-capable System One models (Clef / Clef Flash).

The fixture test_image.png contains the text 'Hello World' and a red circle.
Only Clef and Clef Flash accept the `images` parameter; other models
reject it. `state` is still required and holds the context for the decision.

Run:  python example_image.py [model]     (default: clef:27b)
"""

import base64
import os
import sys

from systemone import SystemOneError, systemone

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "clef:27b"


def load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def main() -> None:
    image_b64 = load_image_b64(os.path.join(HERE, "test_image.png"))

    response = systemone(
        MODEL,
        state="A user shared this image and asks questions about what it shows.",
        questions={
            "has_hello": {
                "type": "noul",
                "instructions": "Does the image contain the word 'Hello'?",
                "criteria": {
                    "true": "The word 'Hello' is visible in the image.",
                    "false": "The word 'Hello' is not visible in the image.",
                },
            },
            "shape": {
                "type": "choice",
                "instructions": "Which shape is drawn in the image?",
                "criteria": {
                    "circle": "A circle or ellipse",
                    "square": "A square or rectangle",
                    "triangle": "A triangle",
                    "none": "No shape is drawn",
                },
            },
            "shape_color": {
                "type": "choice",
                "instructions": "What color is the drawn shape?",
                "criteria": {
                    "red": "The shape is red",
                    "blue": "The shape is blue",
                    "green": "The shape is green",
                    "other": "A different color",
                },
            },
        },
        images=[image_b64],
    )

    print(f"model:   {response.model}")
    print(f"hello:   {response.answers['has_hello'].noul:.4f} (expect ~1.0)")
    shape = response.answers["shape"]
    print(f"shape:   {shape.choice} (confidence {shape.confidence:.4f}, expect 'circle')")
    color = response.answers["shape_color"]
    print(f"color:   {color.choice} (confidence {color.confidence:.4f}, expect 'red')")
    print(f"usage:   {response.usage}")


if __name__ == "__main__":
    try:
        main()
    except SystemOneError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
