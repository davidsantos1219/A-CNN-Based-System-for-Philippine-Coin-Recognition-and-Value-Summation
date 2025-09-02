# coin_recognition_sum.py

import cv2
import numpy as np
import json
import torch
import torchvision.transforms as transforms
from train_coin_classifier import CoinClassifier  # our student-made model


MODEL_FILE        = "coin_classifier.pth"
LABELS_FILE       = "class_to_idx.json"
ROI_SIZE          = 64
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_DISPLAY_WIDTH = 800

# Load the saved label map
with open(LABELS_FILE, "r") as f:
    label_map = json.load(f)
# invert to {idx → label}
idx_to_label = {idx: lbl for lbl, idx in label_map.items()}

# Initialize the model and load weights
model = CoinClassifier(len(label_map)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.eval()

# Define preprocessing for each cropped coin
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((ROI_SIZE, ROI_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def detect_and_sum(image_path):
    # load image
    original_img = cv2.imread(image_path)
    display_img  = original_img.copy()
    gray_img     = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # apply thresholding
    _, binary_img = cv2.threshold(
        gray_img, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # find coin contours
    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    total_sum = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.7:
            continue

        # get ROI with padding
        x, y, w, h = cv2.boundingRect(cnt)
        pad = int(max(w, h) * 0.2)
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + w + pad, original_img.shape[1]), min(y + h + pad, original_img.shape[0])
        roi = original_img[y1:y2, x1:x2]

        # classify the ROI
        input_tensor = preprocess(roi).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(input_tensor)
        pred_idx   = logits.argmax(1).item()
        coin_label = idx_to_label[pred_idx]
        coin_value = int(coin_label.replace("P", ""))
        total_sum += coin_value

        # draw result
        cv2.drawContours(display_img, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(
            display_img, coin_label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 255), 2
        )

    # overlay total
    cv2.putText(
        display_img,
        f"Total: P{total_sum}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (255, 0, 0), 2
    )

    # resize for display (keep aspect ratio)
    h, w = display_img.shape[:2]
    scale = MAX_DISPLAY_WIDTH / float(w)
    disp  = cv2.resize(display_img, (int(w * scale), int(h * scale)))

    cv2.imshow("Detected Coins", disp)
    cv2.imwrite("detection_result.png", display_img)  # save full-res
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python coin_recognition_sum.py path/to/test_image.jpg")
        sys.exit(1)
    detect_and_sum(sys.argv[1])
