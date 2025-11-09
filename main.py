import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
import requests
from io import BytesIO
import argparse


def generate_description(image_path, prompt="a photograph of"):
    """
    Generates a description for the given image, with an optional prompt.
    """
    try:
        if image_path.startswith("http"):
            response = requests.get(image_path)
            response.raise_for_status()  # Raise an exception for bad status codes
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)
    except requests.exceptions.RequestException as e:
        return f"Error: Could not download image from URL. {e}"
    except FileNotFoundError:
        return f"Error: Image file not found at {image_path}"
    except Exception as e:
        return f"Error: Could not open image. {e}"


    # Load the pre-trained model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Preprocess the image and prompt
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    # Generate the description
    outputs = model.generate(**inputs)

    # Decode the description
    description = processor.decode(outputs[0], skip_special_tokens=True)

    # The model often includes the prompt in the output, so we remove it.
    if description.startswith(prompt):
        description = description[len(prompt):].strip()

    return description


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a description for an image.")
    parser.add_argument("image_source", nargs='?', default="profile_picture.jpeg", help="URL or local path of the image.")
    parser.add_argument("--prompt", default="a description of this image:", help="Prompt to guide the description generation.")
    args = parser.parse_args()

    description = generate_description(args.image_source, prompt=args.prompt)
    print(f"Generated Description: {description}")
