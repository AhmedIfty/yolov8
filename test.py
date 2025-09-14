from ultralytics import YOLO
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/yolov8m_refined_v3_tuned.pt")
    p.add_argument("--data",    default="dataset-refined-v3/data.yaml")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--device",  default=0)
    p.add_argument("--source",  default="dataset-refined-v3/test/images")
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--max_det", type=int,   default=300)
    p.add_argument("--project", default=None)
    p.add_argument("--name",    default=None)
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.weights)

    # 1) Validation (with per-class AP table + plots)
    print("\n=== Running Validation (Test Split) ===")
    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        plots=True,        # saves results.png, PR curves, confusion matrix
        conf=args.conf,
        iou=args.iou,
        verbose=True,      # prints per-class AP/P/R in terminal
        project=args.project,
        name=args.name
    )

    # 2) Prediction (save annotated images, no per-image spam)
    print("\n=== Running Prediction (Test Images) ===")
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        show=False,
        verbose=False,     # quiet predict
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name=args.name
    )

if __name__ == "__main__":
    main()
