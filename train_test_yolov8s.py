from ultralytics import YOLO
import os
import torch
import yaml
from datetime import datetime

def main():
    # Paths
    DATA_YAML = 'data.yaml'  # Path to data.yaml (relative to this script)
    MODEL = 'yolov8s.pt'     # Use yolov11 model
    EPOCHS = 100              # Train for more epochs
    IMG_SIZE = 640
    BATCH_SIZE = 16          # Optimal batch size for this dataset
    DEVICE = 0  # Use GPU if available

    # 1. Train the model
    model = YOLO(MODEL)
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,    # Add batch size parameter
        project='runs',
        name='detect_train',
        exist_ok=True,
        device=DEVICE
    )

    # 2. Validate the model
    val_results = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        project='runs',
        name='detect_val',
        exist_ok=True,
        device=DEVICE
    )

    # Print mAP50 (accuracy out of 100) for validation set
    try:
        map50 = val_results.box.map50  # Use .box.map50 instead of .metrics
        print(f"Validation mAP50: {map50*100:.2f} out of 100")
    except Exception as e:
        print("Could not extract mAP50 from validation results.", e)

    # 3. Predict on test images
    test_images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test/images'))
    timestamp = datetime.now().strftime('%Y%m%d')
    predicted_images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f'predicted_images_{timestamp}'))

    predict_results = model.predict(
        source=test_images_dir,
        save=True,
        project=predicted_images_dir,
        name='',  # No subfolder, save directly in predicted_images
        exist_ok=True,
        device=DEVICE
    )

    print('Training, validation, and prediction complete!')
    print(f'Predictions saved to: {predicted_images_dir}')

    # 4. Evaluate accuracy on the test set (requires test/images and test/labels)
    test_yaml = {
        'path': os.path.abspath(os.path.join(os.path.dirname(__file__), 'test')),
        'train': '',
        'val': 'images',
        'nc': 7,
        'names': ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
    }
    test_yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data.yaml'))
    with open(test_yaml_path, 'w') as f:
        yaml.dump(test_yaml, f)

    test_val_results = model.val(
        data=test_yaml_path,
        imgsz=IMG_SIZE,
        project='runs',
        name='detect_test_val',
        exist_ok=True,
        device=DEVICE
    )
    try:
        test_map50 = test_val_results.box.map50  # Use .box.map50 instead of .metrics
        print(f"Test set mAP50: {test_map50*100:.2f} out of 100")
    except Exception as e:
        print("Could not extract mAP50 from test set validation results.", e)

if __name__ == "__main__":
    main()
