from ultralytics import YOLO

if __name__ == '__main__':
    # Option 1: Train from scratch with CAF architecture
    # model = YOLO('ultralytics/cfg/models/v8/yolov8m-caf.yaml')

    # Option 2: Load pretrained weights and modify architecture
    model = YOLO('yolov8m.pt')
    model = YOLO('ultralytics/cfg/models/v8/yolov8m-caf.yaml').load('yolov8m.pt')

    model.train(
        data='dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project='runs/train',
        name='waste-caf-exp1',
        exist_ok=True,
        patience=20,  # Early stopping patience
        save=True,
        save_period=5,  # Save checkpoint every 5 epochs
        val=True,
        plots=True,
        verbose=True
    )