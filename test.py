import os
os.environ["WANDB_DISABLED"] = "true"

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import timm
from torchvision import transforms

# ==========================================
# CONFIG
# ==========================================

MODEL_PATH = "fetal_brain_swin_best.pth"

BG_COLOR = "#0f172a"
CARD_COLOR = "#1e293b"
ACCENT = "#38bdf8"
TEXT = "#f8fafc"
SUCCESS = "#22c55e"

# ==========================================
# LOAD MODEL
# ==========================================

try:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    classes = checkpoint["classes"]

    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

except Exception as e:
    print("MODEL LOAD ERROR:", e)
    exit()

# ==========================================
# IMAGE TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==========================================
# PREDICTION
# ==========================================

def predict_image():

    filepath = filedialog.askopenfilename(
        title="Select Ultrasound Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp")
        ]
    )

    if not filepath:
        return

    try:

        image = Image.open(filepath).convert("RGB")

        display = image.copy()
        display.thumbnail((450, 450))

        tk_img = ImageTk.PhotoImage(display)

        image_label.config(image=tk_img)
        image_label.image = tk_img

        x = transform(image).unsqueeze(0)

        with torch.no_grad():

            outputs = model(x)

            probs = torch.softmax(outputs, dim=1)

            confidence, pred = torch.max(
                probs,
                dim=1
            )

            predicted_class = classes[
                pred.item()
            ]

        predicted_class = predicted_class.replace(
            "_",
            " "
        ).title()

        result_label.config(
            text=predicted_class,
            fg=SUCCESS
        )

        confidence_label.config(
            text=f"Confidence : {confidence.item()*100:.2f}%"
        )

    except Exception as e:

        messagebox.showerror(
            "Prediction Error",
            str(e)
        )

# ==========================================
# WINDOW
# ==========================================

root = tk.Tk()

root.title(
    "AI Fetal Brain Abnormality Detection"
)

root.geometry("1200x800")
root.configure(bg=BG_COLOR)

# ==========================================
# HEADER
# ==========================================

header = tk.Label(
    root,
    text="AI Fetal Brain Abnormality Detection System",
    font=("Segoe UI", 24, "bold"),
    bg=BG_COLOR,
    fg=TEXT
)

header.pack(pady=20)

sub = tk.Label(
    root,
    text="Swin Transformer Based Ultrasound Classification",
    font=("Segoe UI", 12),
    bg=BG_COLOR,
    fg="#94a3b8"
)

sub.pack()

# ==========================================
# MAIN CARD
# ==========================================

card = tk.Frame(
    root,
    bg=CARD_COLOR,
    bd=0
)

card.pack(
    padx=20,
    pady=20,
    fill="both",
    expand=True
)

# ==========================================
# LEFT PANEL
# ==========================================

left = tk.Frame(
    card,
    bg=CARD_COLOR
)

left.pack(
    side="left",
    fill="both",
    expand=True,
    padx=20,
    pady=20
)

image_label = tk.Label(
    left,
    bg=CARD_COLOR
)

image_label.pack()

# ==========================================
# RIGHT PANEL
# ==========================================

right = tk.Frame(
    card,
    bg=CARD_COLOR
)

right.pack(
    side="right",
    fill="both",
    expand=True,
    padx=20,
    pady=20
)

upload_btn = tk.Button(
    right,
    text="Upload Ultrasound Image",
    command=predict_image,
    font=("Segoe UI", 14, "bold"),
    bg=ACCENT,
    fg="black",
    width=25,
    height=2,
    cursor="hand2"
)

upload_btn.pack(pady=30)

title2 = tk.Label(
    right,
    text="Detection Result",
    font=("Segoe UI", 18, "bold"),
    bg=CARD_COLOR,
    fg=TEXT
)

title2.pack(pady=10)

result_label = tk.Label(
    right,
    text="Waiting for Image...",
    font=("Segoe UI", 20, "bold"),
    bg=CARD_COLOR,
    fg=TEXT,
    wraplength=350
)

result_label.pack(pady=20)

confidence_label = tk.Label(
    right,
    text="Confidence : -- %",
    font=("Segoe UI", 16),
    bg=CARD_COLOR,
    fg=TEXT
)

confidence_label.pack()

# ==========================================
# FOOTER
# ==========================================

footer = tk.Label(
    root,
    text="Powered by Swin Transformer | Validation Accuracy : 98.70%",
    font=("Segoe UI", 10),
    bg=BG_COLOR,
    fg="#94a3b8"
)

footer.pack(pady=10)

root.mainloop()