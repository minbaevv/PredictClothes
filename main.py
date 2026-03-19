from fastapi import FastAPI, HTTPException, UploadFile, File
import uvicorn
import torch
import io
from torchvision import transforms
import torch.nn as nn
from PIL import Image

class CheckImage(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10)
        )

    def forward(self, i):
        i = self.first(i)
        i = self.second(i)
        return i

transforms =transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
])
check_image_app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckImage()
torch.load('model.pth', map_location=device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
model.eval()

@check_image_app.post('/predict')
async def check_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=404, detail="Файл кошулган жок")

        img = Image.open(io.BytesIO(image_data))
        img_tensor = transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            pred = y_pred.argmax(dim=1).item()
            mapping = {
                0:'T-shirt/top',
                1:'Trouser',
                2:'Pullover',
                3:'Dress',
                4:'Coat',
                5:'Sandal',
                6:'Shirt',
                7:'Sneaker',
                8:'Bag',
                9:'Ankle boot'

            }

        return {"result": mapping[pred]}


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(check_image_app, host="127.0.0.1", port=8000)