import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.optim import AdamW
import torch

def fine_tune_model(feedback_file='feedback.csv'):
    """
    Fine-tunes the model based on user feedback.
    """
    if not pd.io.common.file_exists(feedback_file):
        print("Feedback file not found. No training will be performed.")
        return

    # Explicitly set the data types for the columns to avoid misinterpretation
    dtype_spec = {'image_path': str, 'user_feedback': str}
    feedback_df = pd.read_csv(feedback_file, dtype=dtype_spec)

    # Drop rows where user_feedback is missing
    feedback_df.dropna(subset=['user_feedback'], inplace=True)

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for index, row in feedback_df.iterrows():
        image_path = row['image_path']
        correct_description = row['user_feedback']

        # Additional check for empty strings
        if not correct_description:
            print(f"Skipping row {index} due to empty feedback.")
            continue

        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Image not found at {image_path}. Skipping this feedback.")
            continue
        except Exception as e:
            print(f"Could not open image {image_path}. Error: {e}. Skipping this feedback.")
            continue

        inputs = processor(images=image, text=correct_description, return_tensors="pt", padding="max_length", truncation=True)
        
        # Set the input_ids as labels for language modeling
        labels = inputs.input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100 # Ignore pad tokens in loss calculation
        
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Processed feedback for {image_path}. Loss: {loss.item()}")

    # Save the fine-tuned model
    model.save_pretrained("./fine-tuned-model")
    processor.save_pretrained("./fine-tuned-model")
    print("Model fine-tuning complete and saved to ./fine-tuned-model")

if __name__ == '__main__':
    fine_tune_model()
