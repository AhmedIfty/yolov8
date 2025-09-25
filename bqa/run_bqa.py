#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BQA = HERE / "bqa_rerank_v5.py"

def main():
    ap = argparse.ArgumentParser(description="BQA launcher (YOLO + CLIP reranker)")
    ap.add_argument("--weights", default="../weights/yolov8m_refined_v4_baseline.pt", help="YOLO weights")
    ap.add_argument("--input", default=str(HERE / "bqa_test"), help="Image or folder")
    ap.add_argument("--output", default=str(HERE / "bqa_result"), help="Output folder")
    ap.add_argument("--query", default=None, help="Text query, e.g., 'syringe'")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--yolo_conf", type=float, default=0.15)
    ap.add_argument("--yolo_iou", type=float, default=0.50)
    ap.add_argument("--topk", type=int, default=75)
    ap.add_argument("--fuse_lambda", type=float, default=0.50)
    ap.add_argument("--fuse_thr", type=float, default=0.50)
    ap.add_argument("--neg_beta", type=float, default=0.0)
    ap.add_argument("--clip", default="ViT-B-32", help="open-clip model name")
    ap.add_argument("--clip_pretrained", default="laion2b_s34b_b79k", help="open-clip pretrained tag")
    args = ap.parse_args()

    query = args.query
    if not query:
        try:
            query = input("Enter query (e.g., 'syringe', 'knife', 'plastic bottle'): ").strip()
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(1)
        if not query:
            print("No query provided. Exiting.")
            sys.exit(1)

    cmd = [
        sys.executable, str(BQA),
        "--weights", args.weights,
        "--input", args.input,
        "--output", args.output,
        "--query", query,
        "--imgsz", str(args.imgsz),
        "--yolo_conf", str(args.yolo_conf),
        "--yolo_iou", str(args.yolo_iou),
        "--topk", str(args.topk),
        "--fuse_lambda", str(args.fuse_lambda),
        "--fuse_thr", str(args.fuse_thr),
        "--neg_beta", str(args.neg_beta),
        "--clip", args.clip,
        "--clip_pretrained", args.clip_pretrained,
    ]

    print("[INFO] Running:", " ".join(map(str, cmd)))
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
