# gradcam_xray.py
# Grad-CAM for your ResNet18 pneumonia model (binary: NORMAL vs PNEUMONIA)
#
# Usage (Windows):
#   python gradcam_xray.py --weights checkpoints/best_resnet18_xray.pt --img "path\to\xray.jpg" --out "gradcam.png"
#
# Notes:
# - This assumes you trained ResNet18 with: model.fc = nn.Linear(..., 1)
# - Output is a heatmap overlay showing what regions most influenced the pneumonia score.

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)  # binary logit
    model.to(device)
    model.eval()
    return model


def get_preprocess():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class GradCAM:
    """
    Minimal Grad-CAM for CNNs.

    For ResNet18, a good target layer is model.layer4[-1].conv2 (last conv).
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Forward hook: save activations
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        # Backward hook: save gradients d(output)/d(activations)
        def bwd_hook(module, grad_in, grad_out):
            # grad_out is a tuple; [0] has same shape as activations
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
          - "pneumonia": use the single output logit (higher => more pneumonia)
          - "normal": use negative logit (higher => more normal) for a "normal CAM"
        """
        self.model.zero_grad(set_to_none=True)

        logit = self.model(x).squeeze(1)  # shape [1]

        if target_logit == "normal":
            score = (-logit).sum()
        else:
            score = logit.sum()

        score.backward()

        # activations: [1, C, H, W]
        # gradients:   [1, C, H, W]
        acts = self.activations
        grads = self.gradients

        if acts is None or grads is None:
            raise RuntimeError("Hooks did not capture activations/gradients. Check target_layer.")

        # Global-average pool gradients over spatial dims -> weights per channel
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted sum of activations across channels
        cam = (weights * acts).sum(dim=1, keepdim=False)  # [1, H, W]
        cam = torch.relu(cam)  # Grad-CAM uses ReLU

        # Normalize to [0,1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()[0], logit.detach().cpu().numpy()[0]


def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


def overlay_and_save(original_pil, cam_01, out_path, alpha=0.45):
    """
    original_pil: original image (any size)
    cam_01: heatmap in [0,1] at 224x224 (from model input)
    """
    # Resize heatmap back to original image size
    cam_img = Image.fromarray((cam_01 * 255).astype(np.uint8)).resize(original_pil.size, resample=Image.BILINEAR)
    cam_arr = np.array(cam_img)

    # Plot overlay
    plt.figure(figsize=(7, 7))
    plt.imshow(original_pil)
    plt.imshow(cam_arr, alpha=alpha, cmap="jet")  # heatmap overlay
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to best_resnet18_xray.pt")
    ap.add_argument("--img", required=True, help="Path to an X-ray image (jpg/png)")
    ap.add_argument("--out", default="gradcam_overlay.png", help="Output image path")
    ap.add_argument("--target", choices=["pneumonia", "normal"], default="pneumonia",
                    help="Which class-style CAM to generate (binary logit model).")
    args = ap.parse_args()

    device = get_device()
    print("Device:", device)

    model = build_model(device)

    # Load your trained weights
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)

    # Choose target layer (last conv layer in ResNet18 block)
    target_layer = model.layer4[-1].conv2
    cammer = GradCAM(model, target_layer)

    try:
        preprocess = get_preprocess()

        orig = load_image(args.img)
        x = preprocess(orig).unsqueeze(0).to(device)  # [1,3,224,224]

        cam_01, logit = cammer(x, target_logit=args.target)
        prob = 1 / (1 + np.exp(-logit))  # sigmoid

        print(f"Raw logit: {logit:.4f}")
        print(f"Pneumonia probability (sigmoid): {prob:.4f}")
        print(f"Saving Grad-CAM overlay -> {args.out}")

        overlay_and_save(orig, cam_01, args.out, alpha=0.45)

    finally:
        cammer.close()


if __name__ == "__main__":
    main()
