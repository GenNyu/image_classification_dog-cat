import argparse
import os
import numpy as np
from PIL import Image
import tensorflow as tf

CLASS_NAMES = ["cat", "dog"] 

def load_image(path, size):
    img = Image.open(path).convert("RGB").resize(size)
    x = np.array(img, dtype=np.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = x[None, ...]   # (1, H, W, 3), 0..255
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Path to a single image.")
    ap.add_argument("--model", type=str, default="models/dogcat_mobilenetv2.keras")
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        raise SystemExit(f"Model not found at: {args.model}")

    model = tf.keras.models.load_model(args.model, compile=False)
    H, W = model.input_shape[1], model.input_shape[2]
    x = load_image(args.image, (W, H))

    y = model.predict(x, verbose=0)
    if y.shape[-1] == 1:
        p = float(y[0, 0])
        idx = int(p >= 0.5)
        conf = p if idx == 1 else 1 - p
    else:
        idx = int(np.argmax(y[0]))
        conf = float(np.max(y[0]))

    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
    print(f"Class: {label}\nConfidence Score: {conf:.3f}")

if __name__ == "__main__":
    main()
