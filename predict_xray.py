import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "checkpoints/best_resnet18_xray.pt"
IMG_SIZE = 224

def main(img_path: str):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(dev)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    model = model.to(dev).eval()

    with torch.no_grad():
        logit = model(x).squeeze(0)
        prob = torch.sigmoid(logit).item()

    pred = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    print(f"Prediction: {pred} | Probability(PNEUMONIA)={prob:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_xray.py path/to/image.jpeg")
        raise SystemExit(1)
    main(sys.argv[1])
