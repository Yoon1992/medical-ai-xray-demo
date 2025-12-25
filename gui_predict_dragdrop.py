import os
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

import torch
import torch.nn as nn
from torchvision import models, transforms

# Drag & Drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    TkinterDnD = None
    DND_FILES = None


# -----------------------------
# CONFIG - EDIT THESE
# -----------------------------
MODEL_PATH = r"checkpoints\best_resnet18_xray.pt"  # <-- change to your .pt path
IMG_SIZE = 224
THRESHOLD = 0.5  # probability threshold for PNEUMONIA

LABEL_NEG = "NORMAL"
LABEL_POS = "PNEUMONIA"
WINDOW_TITLE = "Medical AI Predictor (X-ray)"


def build_model(device: torch.device) -> torch.nn.Module:
    """
    Assumes your trained model is ResNet18 with:
      model.fc = nn.Linear(in_features, 1)
    Output is a single logit (binary classification).
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def make_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("760x520")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Model not found",
                f"MODEL_PATH does not exist:\n{MODEL_PATH}\n\nEdit MODEL_PATH in the script."
            )
            raise SystemExit(1)

        self.model = build_model(self.device)
        self.tfm = make_transform()

        # UI
        self.header = tk.Label(root, text="Drag & drop an X-ray image here, or click Browse",
                               font=("Segoe UI", 14, "bold"))
        self.header.pack(pady=12)

        self.drop_area = tk.Label(
            root,
            text="Drop image here\n(.jpg .jpeg .png .bmp .webp)",
            relief="ridge",
            borderwidth=3,
            width=55,
            height=10,
            font=("Segoe UI", 12)
        )
        self.drop_area.pack(pady=10)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=8)

        self.btn_browse = tk.Button(btn_frame, text="Browse Image", command=self.browse_image,
                                    font=("Segoe UI", 11), width=18)
        self.btn_browse.grid(row=0, column=0, padx=8)

        self.btn_clear = tk.Button(btn_frame, text="Clear", command=self.clear,
                                   font=("Segoe UI", 11), width=10)
        self.btn_clear.grid(row=0, column=1, padx=8)

        # Output
        self.result_var = tk.StringVar(value="Prediction: -")
        self.prob_var = tk.StringVar(value="Probability: -")

        self.result_lbl = tk.Label(root, textvariable=self.result_var, font=("Segoe UI", 16, "bold"))
        self.result_lbl.pack(pady=8)

        self.prob_lbl = tk.Label(root, textvariable=self.prob_var, font=("Segoe UI", 12))
        self.prob_lbl.pack(pady=2)

        # Image preview
        self.preview_lbl = tk.Label(root)
        self.preview_lbl.pack(pady=10)
        self._preview_imgtk = None

        # Enable drag & drop if possible
        if TkinterDnD is not None and hasattr(root, "drop_target_register"):
            self.drop_area.drop_target_register(DND_FILES)
            self.drop_area.dnd_bind("<<Drop>>", self.on_drop)
        else:
            # Not fatal; user can still use Browse
            self.drop_area.config(text="Drag & drop not enabled.\nInstall tkinterdnd2 or use Browse.")

    def clear(self):
        self.result_var.set("Prediction: -")
        self.prob_var.set("Probability: -")
        self.preview_lbl.config(image="")
        self._preview_imgtk = None

    def browse_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.predict(path)

    def on_drop(self, event):
        # event.data may contain {path} or multiple paths
        data = event.data.strip()

        # Windows sometimes wraps paths in braces
        if data.startswith("{") and data.endswith("}"):
            data = data[1:-1]

        # If multiple files dropped, take first
        if " " in data and os.path.exists(data.split(" ")[0]):
            data = data.split(" ")[0]

        if os.path.isfile(data):
            self.predict(data)
        else:
            messagebox.showwarning("Drop error", f"Could not read dropped item:\n{event.data}")

    def predict(self, img_path: str):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Image error", f"Could not open image:\n{img_path}\n\n{e}")
            return

        # Preview (fit to window)
        preview = img.copy()
        preview.thumbnail((340, 340))
        self._preview_imgtk = ImageTk.PhotoImage(preview)
        self.preview_lbl.config(image=self._preview_imgtk)

        # Model inference
        x = self.tfm(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(x).squeeze(0)
            prob = torch.sigmoid(logit).item()

        pred = LABEL_POS if prob >= THRESHOLD else LABEL_NEG
        self.result_var.set(f"Prediction: {pred}")
        self.prob_var.set(f"Probability({LABEL_POS}): {prob:.4f}")

        # Optional: color cue
        if pred == LABEL_POS:
            self.result_lbl.config(fg="red")
        else:
            self.result_lbl.config(fg="green")


def main():
    # If tkinterdnd2 is available, root must be TkinterDnD.Tk()
    if TkinterDnD is not None:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
