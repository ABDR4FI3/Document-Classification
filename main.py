import tensorflow as tf
import numpy as np
from PIL import Image
import os

def load_and_preprocess_image(image_path, target_size=(128, 128)):  # Updated size to 160x160
    """
    Load and preprocess a single image for prediction
    """
    # Load image
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=target_size
    )

    # Convert to array and add batch dimension
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize pixel values
    img_array = img_array / 255.0

    return img_array

def predict_batch(model_path, image_paths, class_names=None, batch_size=32):
    """
    Make predictions for multiple images
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)

    results = []

    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = np.vstack([
            load_and_preprocess_image(path)
            for path in batch_paths
        ])

        # Get predictions for batch
        predictions = model.predict(batch_images, verbose=0)  # Added verbose=0 to reduce output

        for j, pred in enumerate(predictions):
            pred_class_index = np.argmax(pred)
            confidence = pred[pred_class_index]

            if class_names:
                pred_class = class_names[pred_class_index]
                results.append((batch_paths[j], pred_class, confidence))
            else:
                results.append((batch_paths[j], pred_class_index, confidence))

    return results

if __name__ == "__main__":
    # Path to your saved model
    MODEL_PATH = '/content/drive/MyDrive/keras/best_model_vol2.keras'

    # Directory containing validation images
    VALIDATION_DIR = '/content/drive/MyDrive/validation'

    # Class names in order
    CLASS_NAMES = [
        'ADVE',
        'Email',
        'Form',
        'Letter',
        'Memo',
        'News',
        'Note',
        'Report',
        'Resume',
        'Scientific',
    ]

    try:
        # Get all image files from the directory
        image_paths = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')

        for filename in os.listdir(VALIDATION_DIR):
            if filename.endswith(valid_extensions):
                full_path = os.path.join(VALIDATION_DIR, filename)
                image_paths.append(full_path)

        if not image_paths:
            raise ValueError(f"No valid images found in {VALIDATION_DIR}")

        print(f"Found {len(image_paths)} images to process")

        # Make predictions
        results = predict_batch(MODEL_PATH, image_paths, CLASS_NAMES)

        # Print results
        print("\nPrediction Results:")
        print("-" * 50)
        for image_path, pred_class, conf in results:
            filename = os.path.basename(image_path)
            print(f"Image: {filename}")
            print(f"Predicted Class: {pred_class}")
            print(f"Confidence: {conf:.2%}")
            print("-" * 50)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")