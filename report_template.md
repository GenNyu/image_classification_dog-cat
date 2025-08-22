
# Short Report — Dog vs Cat Classifier

## 1. Problem & Dataset
- Task: binary classification (dog vs cat) from images.
- Dataset: source(s), size, train/val split, any filtering/cleaning you did.

## 2. Approach
- Model: MobileNetV2 (transfer learning).
- Preprocessing: resizing to 224×224, MobileNetV2 `preprocess_input`.
- Augmentations: flip, light rotation, zoom.
- Loss/metrics: binary cross-entropy, accuracy, AUC.
- Fine-tuning strategy (if any).

## 3. Training Setup
- Hardware (CPU/GPU), training time, batch size, epochs.
- Best validation metrics.

## 4. Results
- Validation accuracy/AUC.
- Example predictions (screenshots or console output).
- Failure cases & error analysis (optional).

## 5. Limitations & Ideas for Improvement
- More diverse data, stronger augmentations, larger models.
- Class imbalance handling, threshold tuning.
- Distillation / quantization / on-device (TFLite) (optional).

## 6. How to Reproduce
- Environment (Python/TensorFlow versions).
- Commands to run (copy from README).
