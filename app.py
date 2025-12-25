import os
import uuid

from flask import Flask, render_template, request
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # for servers / no GUI
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
WEIGHTS_PATH = os.path.join("checkpoints", "best_resnet18_xray.pt")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "results")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(device):
    # ResNet18 with single-logit output
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    model.eval()

    # Load your trained weights
    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    return model


def get_preprocess():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class GradCAM:
    """
    Minimal Grad-CAM for ResNet18.
    Uses the last conv layer in layer4.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._h1 = target_layer.register_forward_hook(fwd_hook)
        self._h2 = target_layer.register_full_backward_hook(bwd_hook)

    def close(self):
        self._h1.remove()
        self._h2.remove()

    def __call__(self, x, target_logit="pneumonia"):
        """
        x: tensor [1,3,224,224]
        target_logit:
          "pneumonia": use the raw logit (higher => more pneumonia)
          "normal":    flip sign for a 'normal' heatmap if needed
        """
        self.model.zero_grad(set_to_none=True)

        logit = self.model(x).squeeze(1)  # [1]

        if target_logit == "normal":
            score = (-logit).sum()
        else:
            score = logit.sum()

        score.backward()

        acts = self.activations      # [1, C, H, W]
        grads = self.gradients       # [1, C, H, W]

        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM hooks did not fire. Check target_layer.")

        # Global-average pool gradients over spatial dimensions
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * acts).sum(dim=1, keepdim=False)  # [1, H, W]
        cam = torch.relu(cam)

        # Normalize to [0,1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()[0], float(logit.detach().cpu().numpy()[0])


# Load model once at startup
device = get_device()
model = build_model(device)
preprocess = get_preprocess()
target_layer = model.layer4[-1].conv2
cammer = GradCAM(model, target_layer)


def make_gradcam_overlay(orig_pil, cam_01, outfile):
    """
    orig_pil: original PIL.Image
    cam_01: 2D numpy array [H,W] in [0,1]
    outfile: path to save PNG
    """
    # Resize CAM to original size
    cam_img = Image.fromarray((cam_01 * 255).astype(np.uint8)).resize(
        orig_pil.size, resample=Image.BILINEAR
    )
    cam_arr = np.array(cam_img)

    plt.figure(figsize=(6, 6))
    plt.imshow(orig_pil, cmap="gray")
    plt.imshow(cam_arr, alpha=0.45, cmap="jet")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


@app.route("/", methods=["GET", "POST"])
def index():
    prob = None
    heatmap_filename = None
    raw_logit = None
    error_msg = None

    if request.method == "POST":
        if "file" not in request.files:
            error_msg = "No file part in request."
        else:
            f = request.files["file"]
            if f.filename == "":
                error_msg = "No file selected."
            else:
                try:
                    # Load image
                    orig = Image.open(f.stream).convert("RGB")
                    x = preprocess(orig).unsqueeze(0).to(device)  # [1,3,224,224]

                    # Grad-CAM
                    cam_01, logit = cammer(x, target_logit="pneumonia")
                    raw_logit = logit
                    prob = 1.0 / (1.0 + np.exp(-logit))  # sigmoid

                    # Save heatmap overlay
                    uid = str(uuid.uuid4())[:8]
                    heatmap_filename = f"gradcam_{uid}.png"
                    out_path = os.path.join(app.config["UPLOAD_FOLDER"], heatmap_filename)
                    make_gradcam_overlay(orig, cam_01, out_path)

                except Exception as e:
                    error_msg = f"Error processing image: {e}"

    return render_template(
        "index.html",
        prob=prob,
        raw_logit=raw_logit,
        heatmap_filename=heatmap_filename,
        error_msg=error_msg,
    )


@app.route("/about")
def about():
    # you can expand later for explanation / disclaimer
    return "Medical AI demo - not for clinical use."


@app.teardown_appcontext
def cleanup(exception=None):
    # If you ever want to dispose hooks / model, could do here
    pass


if __name__ == "__main__":
    # For local testing
    app.run(debug=True)
