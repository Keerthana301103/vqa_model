import streamlit as st
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch

# Load the model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return processor, model

processor, model = load_model()

# Streamlit UI
st.title("Visual Question Answering (VQA) with BLIP 🤖🖼️")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
question = st.text_input("Enter your question about the image:")

if uploaded_file and question:
    image = Image.open(uploaded_file).convert('RGB')
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown(f"**Q:** {question}")
    st.markdown(f"**A:** {answer}")
