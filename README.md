
# Dog vs Cat Classifier (TensorFlow/Keras)

A simple image classification project to distinguish **dogs** vs **cats**.  
Meets the assignment requirements:
- Program takes an **image file** as input and **prints** the predicted label.
- Includes a brief **report template** and **README** to reproduce results.
- You can **train** locally or export a model from **Teachable Machine** and use `predict.py`.

---

## 1) Environment

```bash
# (Option A) Create a virtual env (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\activate

# (Option B) macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> If you have a GPU, install a GPU build of TensorFlow following the official docs (optional).

---

## 2) Prepare data

Organize your images into the following structure (use any dog/cat images you collect or a Kaggle subset):

```
data/
  train/
    cat/
      xxx.jpg, yyy.png, ...
    dog/
      aaa.jpg, bbb.png, ...
  val/
    cat/
    dog/
# (optional)
  test/
    cat/
    dog/
```

- Recommended: at least a few hundred images per class if possible.
- Images can be JPG/PNG; non-images will be ignored.

---

## 3) Train

```bash
python train.py --data_dir data --epochs 10 --model_out models/dogcat_mobilenetv2.keras
```
The script will:
- Build a MobileNetV2 transfer-learning model
- Use simple data augmentation
- Save the **best model** to `models/dogcat_mobilenetv2.keras`
- Save class names to `models/class_names.txt`

**Fine‑tuning (optional):**
```bash
python train.py --data_dir data --epochs 5 --fine_tune_from -40 --lr 1e-5
```

---

## 4) Predict on a single image

```bash
python predict.py --image path/to/your_image.jpg --model models/dogcat_mobilenetv2.keras
```
Example output:
```
dog (0.973)
```

- If you trained using different class names or order, adjust `models/class_names.txt` accordingly.
- If you use **Teachable Machine** (see below), export a **Keras (.h5 or .keras)** model and place it at `models/dogcat_mobilenetv2.keras`. Then create `models/class_names.txt` with two lines, e.g.:
  ```
  cat
  dog
  ```

---

## 5) Use a Teachable Machine model (no-code training)

1. Go to Teachable Machine → Image Project → add **Dog** and **Cat** classes → train.
2. Export → **TensorFlow → Keras** → Download model. You’ll get a Keras model file.
3. Put the model file at `models/dogcat_mobilenetv2.keras`
4. Create `models/class_names.txt` listing the two class names in the same order you used.
5. Run prediction:
   ```bash
   python predict.py --image path/to/img.jpg --model models/dogcat_mobilenetv2.keras
   ```

---

## 6) Evaluate (optional)

If you prepared a `data/test` folder like the train/val structure:
```bash
python predict.py --folder data/test --model models/dogcat_mobilenetv2.keras
```
This prints per-image predictions and an overall accuracy if labels are inferable from parent folders.

---

## 7) Files

- `train.py` — Training script (TensorFlow/Keras + MobileNetV2 transfer learning)
- `predict.py` — Inference script for single image or a folder
- `requirements.txt` — Python dependencies
- `report_template.md` — Template for the short report you’ll submit

Good luck!
