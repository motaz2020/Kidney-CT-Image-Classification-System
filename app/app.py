from fastapi import FastAPI, UploadFile
import torch
from PIL import Image
import torchvision.transforms as transforms
from src.model import load_model 

app = FastAPI()

model = load_model("models/best_model.pth")
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile):
    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        pred = outputs.argmax(dim=1).item()

    return {"prediction": pred}