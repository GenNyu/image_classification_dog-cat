import argparse
import os
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

AUTOTUNE = tf.data.AUTOTUNE

def build_datasets(data_dir, img_size=224, batch_size=32):
    # Trích xuất dữ liệu
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        raise SystemExit(f"Need {data_dir}/train and {data_dir}/val with class subfolders.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(img_size, img_size), batch_size=batch_size, shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=(img_size, img_size), batch_size=batch_size, shuffle=False
    )

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    def pp(image, label):
        image = tf.cast(image, tf.float32)
        image = preprocess(image)
        return image, label

    data_augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ])

    def aug(image, label):
        image = data_augment(image)
        return image, label

    train_ds = train_ds.map(aug, num_parallel_calls=AUTOTUNE).map(pp, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(pp, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds

def build_model(img_size=224, num_classes=2, fine_tune_from=None):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x) # Binary classification

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),
                 tf.keras.metrics.AUC(name="auc")]
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--model_out", type=str, default="models/dogcat_mobilenetv2.keras")
    args = parser.parse_args()

    train_ds, val_ds = build_datasets(args.data_dir, args.img_size, args.batch_size)
    model = build_model(args.img_size)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model_out,
            monitor="val_auc",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save(args.model_out)
    print(f"Saved best model to {args.model_out}")

if __name__ == "__main__":
    main()
