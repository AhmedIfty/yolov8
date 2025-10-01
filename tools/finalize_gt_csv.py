# save as tools/finalize_gt_csv.py
import csv, os

IN_CSV  = "../bqa-evaluation/gt_template.csv"   # or gt_subtypes.csv if you edited that
OUT_CSV = "../bqa-evaluation/gt_subtypes.csv"

CANON = {
    "knife":"knife","chef knife":"knife","kitchen knife":"knife","butter knife":"knife",
    "fork":"fork",
    "scissor":"scissors","scissors":"scissors","shears":"scissors",
    "razor":"razor","razor blade":"razor","safety razor":"razor","double edged razor":"razor","blade":"razor",
    "nail":"nail","metal nail":"nail","construction nail":"nail",
    "pin":"pin","thumbtack":"pin","push pin":"pin","tack":"pin","safety pin":"pin",
    "syringe":"syringe","test tube":"test tube","mask":"mask","glove":"glove"
}

def canonize(s):
    t = (s or "").strip().lower()
    return CANON.get(t, t)

def main():
    rows_out = []
    with open(IN_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sub = canonize(row["subtype"])
            if not sub:
                continue  # ignore unlabeled
            row["subtype"] = sub
            rows_out.append(row)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["crop_name","crop_name_flat","image","x1","y1","x2","y2","superclass","subtype"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {len(rows_out)} labeled rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
