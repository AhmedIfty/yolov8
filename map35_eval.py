import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def yolo_to_coco_gt(images_dir, yolo_labels_dir, class_names):
    coco_gt = {
        "info": {"description": "COCO-style ground truth for mAP calculation"},
        "licenses": [{"id": 1, "name": "N/A"}],
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)],
    }
    filename_to_id = {}
    ann_id = 1
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    for idx, img_file in enumerate(tqdm(image_files, desc="Generating COCO Ground Truth")):
        img_id = idx + 1
        filename_to_id[img_file] = img_id
        img_path = Path(images_dir) / img_file
        label_path = Path(yolo_labels_dir) / (img_path.stem + ".txt")
        w, h = Image.open(img_path).size
        coco_gt["images"].append({"id": img_id, "file_name": img_file, "width": w, "height": h})

        if not label_path.exists():
            continue

        with open(label_path, "r") as f:
            for line in f:
                cls, x_center, y_center, width, height = map(float, line.strip().split())
                x_min = (x_center - width / 2) * w
                y_min = (y_center - height / 2) * h
                bbox_w = width * w
                bbox_h = height * h
                coco_gt["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": int(cls),
                    "bbox": [x_min, y_min, bbox_w, bbox_h], "area": bbox_w * bbox_h, "iscrowd": 0,
                })
                ann_id += 1
    return coco_gt, filename_to_id


def run_eval():
    root = Path("test-mAP35")
    images_dir = root / "dataset-640/test/images"
    yolo_labels_dir = root / "dataset-640/test/labels"
    weights = root / "weights/yolov8m-baseline-640.pt"
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    class_names = ['glass', 'medical', 'metal', 'organic', 'paper', 'plastic', 'sharp-object']

    gt_json_path = results_dir / "ground_truth.json"
    coco_gt_dict, filename_to_id = yolo_to_coco_gt(images_dir, yolo_labels_dir, class_names)
    with open(gt_json_path, "w") as f:
        json.dump(coco_gt_dict, f, indent=4)
    print(f"\n✅ Ground truth saved to {gt_json_path}")

    model = YOLO(str(weights))
    coco_predictions = []
    sorted_image_files = sorted(filename_to_id.keys())

    for img_file in tqdm(sorted_image_files, desc="Running Model Predictions"):
        img_id = filename_to_id[img_file]
        img_path = images_dir / img_file
        preds = model.predict(img_path, verbose=False)

        for p in preds:
            for box in p.boxes:
                # --- THIS IS THE CORRECTED PART ---
                # Get box in [x_min, y_min, x_max, y_max] format
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                # Convert to COCO's [x_min, y_min, width, height] format
                coco_bbox = [x1, y1, x2 - x1, y2 - y1]
                # --- End of correction ---

                coco_predictions.append({
                    'image_id': img_id,
                    'category_id': int(box.cls),
                    'bbox': coco_bbox,  # Use the correctly formatted bbox
                    'score': float(box.conf),
                })

    pred_json_path = results_dir / "predictions.json"
    with open(pred_json_path, "w") as f:
        json.dump(coco_predictions, f, indent=4)
    print(f"✅ Predictions saved to {pred_json_path}")

    print("\nCalculating mAP @ IoU=0.35...")
    coco_gt = COCO(str(gt_json_path))
    coco_pred = coco_gt.loadRes(str(pred_json_path))

    coco_eval = COCOeval(coco_gt, coco_pred, iouType="bbox")
    coco_eval.params.iouThrs = [0.35]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()  # You can use summarize again, it will work now

    # Or access the stat directly
    map_35 = coco_eval.stats[0]
    print(f"\n✅ Final mAP@0.35 = {map_35:.4f}")


if __name__ == "__main__":
    run_eval()



# import os
# import json
# from pathlib import Path
# from PIL import Image
# from tqdm import tqdm
#
# from ultralytics import YOLO
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
#
#
# def yolo_to_coco_gt(images_dir, yolo_labels_dir, class_names):
#     """
#     Generates a COCO-formatted ground-truth JSON from YOLO labels.
#
#     Returns:
#         A tuple of (COCO dictionary, mapping of filename to image_id).
#     """
#     coco_gt = {
#         "images": [],
#         "annotations": [],
#         "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)],
#     }
#     filename_to_id = {}
#     ann_id = 1
#
#     # Get a deterministically sorted list of image files
#     image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
#
#     for idx, img_file in enumerate(tqdm(image_files, desc="Generating COCO Ground Truth")):
#         img_id = idx + 1
#         filename_to_id[img_file] = img_id
#
#         img_path = Path(images_dir) / img_file
#         label_path = Path(yolo_labels_dir) / (img_path.stem + ".txt")
#
#         w, h = Image.open(img_path).size
#         coco_gt["images"].append({"id": img_id, "file_name": img_file, "width": w, "height": h})
#
#         if not label_path.exists():
#             continue
#
#         with open(label_path, "r") as f:
#             for line in f:
#                 cls, x_center, y_center, width, height = map(float, line.strip().split())
#                 x_min = (x_center - width / 2) * w
#                 y_min = (y_center - height / 2) * h
#                 bbox_w = width * w
#                 bbox_h = height * h
#
#                 coco_gt["annotations"].append({
#                     "id": ann_id,
#                     "image_id": img_id,
#                     "category_id": int(cls),
#                     "bbox": [x_min, y_min, bbox_w, bbox_h],
#                     "area": bbox_w * bbox_h,
#                     "iscrowd": 0,
#                 })
#                 ann_id += 1
#
#     return coco_gt, filename_to_id
#
#
# def run_eval():
#     # --- 1. Define Paths and Parameters ---
#     root = Path("test-mAP35")
#     data_yaml = root / "dataset-640/data.yaml"
#     weights = root / "weights/yolov8m-baseline-640.pt"
#     images_dir = root / "dataset-640/test/images"
#     yolo_labels_dir = root / "dataset-640/test/labels"
#     results_dir = root / "results"
#     results_dir.mkdir(exist_ok=True)
#
#     # These must match the order in your data.yaml
#     class_names = ['glass', 'medical', 'metal', 'organic', 'paper', 'plastic', 'sharp-object']
#
#     # --- 2. Generate Ground Truth COCO file ---
#     gt_json_path = results_dir / "ground_truth.json"
#     coco_gt_dict, filename_to_id = yolo_to_coco_gt(images_dir, yolo_labels_dir, class_names)
#     with open(gt_json_path, "w") as f:
#         json.dump(coco_gt_dict, f, indent=4)
#     print(f"\n✅ Ground truth saved to {gt_json_path}")
#
#     # --- 3. Run Predictions and Generate Aligned Predictions JSON ---
#     model = YOLO(str(weights))
#     coco_predictions = []
#
#     # Iterate through the same sorted file list to guarantee alignment
#     sorted_image_files = sorted(filename_to_id.keys())
#     for img_file in tqdm(sorted_image_files, desc="Running Model Predictions"):
#         img_id = filename_to_id[img_file]
#         img_path = images_dir / img_file
#
#         # Run prediction
#         preds = model.predict(img_path, verbose=False)
#
#         # Format results into COCO prediction format
#         for p in preds:
#             for box in p.boxes:
#                 coco_predictions.append({
#                     'image_id': img_id,
#                     'category_id': int(box.cls),
#                     'bbox': box.xywh.tolist()[0],
#                     # [x_center, y_center, width, height] -> [x_min, y_min, width, height]
#                     'score': float(box.conf),
#                 })
#
#     pred_json_path = results_dir / "predictions.json"
#     with open(pred_json_path, "w") as f:
#         json.dump(coco_predictions, f, indent=4)
#     print(f"✅ Predictions saved to {pred_json_path}")
#
#     # --- 4. Evaluate at IoU=0.35 using pycocotools ---
#     print("\nCalculating mAP @ IoU=0.35...")
#     coco_gt = COCO(str(gt_json_path))
#     coco_pred = coco_gt.loadRes(str(pred_json_path))
#
#     coco_eval = COCOeval(coco_gt, coco_pred, iouType="bbox")
#     coco_eval.params.iouThrs = [0.35]
#
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()
#
#     map_35 = coco_eval.stats[0]
#     print(f"\n✅ Final mAP@0.35 = {map_35:.4f}")
#
#
# if __name__ == "__main__":
#     run_eval()


# import os
# import json
# from pathlib import Path
# from ultralytics import YOLO
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
#
#
# # -------- STEP 1: Convert YOLO labels -> COCO JSON --------
# def yolo_to_coco(yolo_labels_dir, images_dir, output_json, class_names):
#     coco = {
#         "info": {
#             "description": "YOLO dataset converted to COCO format",
#             "version": "1.0",
#             "year": 2025,
#         },
#         "licenses": [
#             {
#                 "id": 1,
#                 "name": "Unknown",
#                 "url": "N/A"
#             }
#         ],
#         "images": [],
#         "annotations": [],
#         "categories": []
#     }
#     ann_id = 1
#
#     # Categories
#     for i, name in enumerate(class_names):
#         coco["categories"].append({"id": i, "name": name})
#
#     # Images + Annotations
#     for idx, img_file in enumerate(sorted(os.listdir(images_dir))):
#         if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue
#
#         img_id = idx + 1
#         img_path = Path(images_dir) / img_file
#         label_path = Path(yolo_labels_dir) / (Path(img_file).stem + ".txt")
#
#         from PIL import Image
#         w, h = Image.open(img_path).size
#         coco["images"].append({
#             "id": img_id,
#             "file_name": img_file,
#             "width": w,
#             "height": h
#         })
#
#         if label_path.exists():
#             with open(label_path, "r") as f:
#                 for line in f.readlines():
#                     cls, x, y, bw, bh = map(float, line.strip().split())
#                     cls = int(cls)
#
#                     # Convert YOLO xywh (normalized) -> COCO xywh (absolute)
#                     x1 = (x - bw / 2) * w
#                     y1 = (y - bh / 2) * h
#                     bw_abs = bw * w
#                     bh_abs = bh * h
#
#                     coco["annotations"].append({
#                         "id": ann_id,
#                         "image_id": img_id,
#                         "category_id": cls,
#                         "bbox": [x1, y1, bw_abs, bh_abs],
#                         "area": bw_abs * bh_abs,
#                         "iscrowd": 0
#                     })
#                     ann_id += 1
#
#     with open(output_json, "w") as f:
#         json.dump(coco, f)
#     print(f"✅ COCO annotations saved to {output_json}")
#
#
# # -------- STEP 2 + 3: Run YOLO val & compute mAP@0.35 --------
# def run_eval():
#     root = Path("test-mAP35")
#     data_yaml = root / "dataset-640/data.yaml"
#     weights = root / "weights/yolov8m-baseline-640.pt"
#     results_dir = root / "results"
#     results_dir.mkdir(exist_ok=True)
#
#     coco_gt_json = results_dir / "test_coco.json"
#     coco_pred_json = results_dir / "predictions.json"
#
#     # Class names (must match your data.yaml order)
#     class_names = ['glass', 'medical', 'metal', 'organic', 'paper', 'plastic', 'sharp-object']
#
#     # Step 1: Convert YOLO txt -> COCO json
#     yolo_to_coco(
#         yolo_labels_dir=root / "dataset-640/test/labels",
#         images_dir=root / "dataset-640/test/images",
#         output_json=coco_gt_json,
#         class_names=class_names
#     )
#
#     # Step 2: Run YOLO validation on test set (saves predictions.json)
#     model = YOLO(str(weights))
#     model.val(
#         data=str(data_yaml),
#         split="test",
#         save_json=True,
#         project=str(results_dir),
#         name="val",
#         exist_ok=True
#     )
#
#     # YOLO saves predictions.json inside results_dir/"val"/
#     pred_file = results_dir / "val/predictions.json"
#     if not pred_file.exists():
#         raise FileNotFoundError(f"Predictions not found at {pred_file}")
#     os.rename(pred_file, coco_pred_json)  # move to results root
#
#     # Step 3: Evaluate at IoU=0.35
#     coco_gt = COCO(str(coco_gt_json))
#     coco_pred = coco_gt.loadRes(str(coco_pred_json))
#     coco_eval = COCOeval(coco_gt, coco_pred, iouType="bbox")
#     coco_eval.params.iouThrs = [0.35]  # only IoU=0.35
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()
#
#
# if __name__ == "__main__":
#     run_eval()
