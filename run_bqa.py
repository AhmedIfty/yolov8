import subprocess

def main():
    # Path to your rerank script and baseline weights
    script = "bqa/bqa_rerank_v3.py"
    weights = "weights/yolov8m_refined_v4_baseline.pt"

    print("\n=== BQA Launcher ===")
    print("Type your query (e.g., 'syringe', 'knife', 'plastic bottle')\n")

    query = input("Enter query: ").strip()
    if not query:
        print("No query entered. Exiting.")
        return

    # Build the command
    cmd = [
        "python", script,
        "--weights", weights,
        "--query", query
    ]

    print("\n[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()