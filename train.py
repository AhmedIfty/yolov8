from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')
    model.train(
        data='dataset-refined-640-v2/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project='runs/train',
        name='refined-exp2',
        exist_ok=True
    )

