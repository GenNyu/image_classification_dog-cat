import argparse, numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

def read_labels(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return [(ln.strip().split(maxsplit=1)+[""])[1] for ln in f if ln.strip()]
    except:
        return None

def load_image(path, size):
    img = Image.open(path).convert("RGB")
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)  
    x = np.asarray(img, dtype=np.float32)
    x = x/127.5 - 1.0                                       
    return x[None, ...]                                     

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Path to a single image.")
    ap.add_argument("--model", default="models/keras_model.h5")
    ap.add_argument("--labels", default="models/labels.txt")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    H, W = model.input_shape[1] or 224, model.input_shape[2] or 224
    x = load_image(args.image, (W, H))
    y = model.predict(x, verbose=0)

    if y.shape[-1] == 1:
        p = float(y[0,0]); idx = int(p >= 0.5); conf = p if idx==1 else 1-p
    else:
        idx = int(np.argmax(y[0])); conf = float(np.max(y[0]))

    labels = read_labels(args.labels)
    label = labels[idx] if labels and idx < len(labels) else str(idx)
    print(f"Class: {label}\nConfidence Score: {conf:.3f}")

if __name__ == "__main__":
    main()
