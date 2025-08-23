# Dog vs Cat Classifier (TensorFlow/Keras)

A simple image classification project to distinguish **dogs** vs **cats**.

---

## Project Tree

```
./
├─ README.md
├─ requirements.txt
├─ .git/
├─ .gitignore
├─ data/
│  ├─ train/
│  └─ val/
├─ models/
│  ├─ dogcat\_mobilenetv2.keras
│  ├─ keras\_model.h5
│  └─ labels.txt
├─ train.py
├─ predict.py
└─ predict\_tm.py
```

---

## Files

* `train.py` — Training script (TensorFlow/Keras + MobileNetV2 transfer learning)
* `predict.py` — Inference script for a single image using the self-trained model
* `predict_tm.py` — Inference script for a single image using the Teachable Machine model
* `requirements.txt` — Python dependencies

---

## How to Run

1. Create a virtual environment & install dependencies

```bash
python -m venv .venv

# Windows (Cmd)
.venv\\Scripts\\activate

pip install -r requirements.txt
```

2. Prepare data with the following structure

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
```

3. Train

```bash
python train.py --data_dir data --epochs 10 --model_out models/dogcat_mobilenetv2.keras
```

4. Predict

* Using the **self-trained model**:

```bash
python predict.py --image path/to/img.jpg
```

* Using the **Teachable Machine** model:

```bash
python predict_tm.py --image path/to/img.jpg
```

**Example output:**

```
Class: dog
Confidence Score: 0.973
```

---

## Deploy with Streamlit

* The Streamlit library is included in requirements.txt.
* Make sure app.py exists (or change the command to your actual entry file).

```bash
streamlit run app.py
```

Your browser will open the UI:
* Choose between two model options: Self-trained and Teachable Machine.
* Upload/select an image to run prediction.
* The app will display the predicted class and confidence.