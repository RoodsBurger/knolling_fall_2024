from ultralytics import YOLO
import os
import random
from pathlib import Path

# Load the model
model = YOLO("./runs/detect/yolo_knolling/weights/best.pt")


for directory in ['VAE_1118_obj2', 'VAE_1118_obj3', 'VAE_1118_obj4', 'VAE_1118_obj5', 
                  'VAE_1118_obj6', 'VAE_1118_obj7', 'VAE_1118_obj8']:

    # Define base paths
    messy_dir = Path(f"../../../../../Desktop/data/dataset/{directory}/origin_images_before")
    tidy_dir = Path(f"../../../../../Desktop/data/dataset/{directory}/origin_images_after")

    # Get all images ending with _0.png from messy directory
    messy_images = [f for f in os.listdir(messy_dir) if f.endswith('_0.png')]

    # Randomly sample 10 image names
    sampled_names = random.sample(messy_images, 10)

    # Process each pair of images
    for name in sampled_names:
        # Process messy image
        messy_path = str(messy_dir / name)
        model.predict(messy_path, save=True, imgsz=640, conf=0.7, 
                     project='predictions', name=f'messy_{directory}_{name}')
        
        # Process tidy image
        tidy_path = str(tidy_dir / name)
        model.predict(tidy_path, save=True, augment=True, visualize=True, imgsz=640, conf=0.7, 
                     project='predictions', name=f'tidy_{directory}_{name}')

print("Predictions saved in predictions directory")
