import streamlit as st
from PIL import Image
import main
import pandas as pd
import os
import train

st.set_page_config(layout="wide", page_title="Image Description Generator")

st.title("Image Description Generator")

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load model and processor
@st.cache_resource
def load_model():
    if os.path.exists("./fine-tuned-model"):
        return main.BlipProcessor.from_pretrained("./fine-tuned-model"), \
               main.BlipForConditionalGeneration.from_pretrained("./fine-tuned-model")
    else:
        return main.BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"), \
               main.BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

processor, model = load_model()

def generate_description(image, prompt):
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    description = processor.decode(outputs[0], skip_special_tokens=True)
    if description.startswith(prompt):
        description = description[len(prompt):].strip()
    return description

# Image Upload
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Create and display a thumbnail
    thumbnail = image.copy()
    thumbnail.thumbnail((200, 200))
    st.image(thumbnail, caption="Uploaded Image")

    prompt = st.text_input("Enter a prompt for the description:", "a description of this image:")

    if st.button("Generate Description"):
        with st.spinner("Generating description..."):
            description = generate_description(image, prompt)
            st.session_state.description = description

    if 'description' in st.session_state:
        st.write("### Generated Description")
        st.write(st.session_state.description)

        st.write("### Provide Feedback")
        feedback = st.text_area("If the description is not accurate, please provide a corrected version:")
        if st.button("Submit Feedback"):
            # Save the uploaded image
            image_path = os.path.join('uploads', uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            feedback_data = {
                'image_path': image_path,
                'generated_description': st.session_state.description,
                'user_feedback': feedback
            }
            
            if not os.path.exists('feedback.csv'):
                df = pd.DataFrame([feedback_data])
                df.to_csv('feedback.csv', index=False)
            else:
                df = pd.read_csv('feedback.csv')
                new_df = pd.DataFrame([feedback_data])
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_csv('feedback.csv', index=False)
            
            st.success("Thank you for your feedback!")

if st.button("Fine-tune Model"):
    with st.spinner("Fine-tuning the model... This may take a while."):
        train.fine_tune_model()
    st.success("Model fine-tuning complete!")
    # Clear the cache to reload the fine-tuned model
    st.cache_resource.clear()
