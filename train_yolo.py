from ultralytics import YOLO

def train_yolo():
    # Load a model
    model = YOLO('yolo11n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device='',
        degrees=180,
        shear=15,
        flipud=0.3,
        fliplr=0.3,
        translate=0.1,
        name='yolo_knolling',  # name of the experiment
        patience=50,                   # early stopping patience
        save=True,                     # save checkpoints
        save_period=5,               # save every 10 epochs
        cache=True,                   # cache images for faster training
        exist_ok=True,                # overwrite existing experiment
        pretrained=True,              # use pretrained model
        optimizer='auto',             # optimizer (SGD, Adam, etc.)
        verbose=True,                 # print verbose output
        seed=42                       # random seed for reproducibility
    )
    
    return results

if __name__ == "__main__":
    results = train_yolo()
    # Print training results
    print(results)
    
    # Optional: Validate the model after training
    model = YOLO('runs/detect/yolo_knolling/weights/best.pt')
    metrics = model.val()
    print(f"Validation metrics: {metrics}")
