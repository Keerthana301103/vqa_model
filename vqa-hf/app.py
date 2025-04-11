from fastapi import FastAPI, UploadFile, File, Form
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import io

app = FastAPI()

# Load BLIP VQA model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

@app.post("/predict")
async def predict(file: UploadFile = File(...), question: str = Form(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    
    return {"question": question, "answer": answer}

# Health check route for Render
@app.get("/")
def read_root():
    return {"message": "Visual Question Answering API is running!"}
