from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')
    model.train(
        data='dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project='runs/train',
        name='waste-exp1-riskaware',
        exist_ok=True
    )