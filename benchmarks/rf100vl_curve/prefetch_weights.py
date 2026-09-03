"""Pre-download every pretrained weight the YOLO and RF-DETR legs need, so no
training unit ever stalls on (or fails from) a first-use network fetch.

  - YOLO:    yolo26{n,s,m,l}.pt  (Ultralytics downloads to CWD on first YOLO(id))
  - RF-DETR: each RFDETR{Nano,Small,Medium,Large}() pulls its COCO-pretrained base

Run once, from this directory, in the same env the benchmark will use.
Idempotent: anything already cached is skipped by the libraries themselves.
"""
from __future__ import annotations

import sys

YOLO_IDS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt"]


def prefetch_yolo() -> None:
    from ultralytics import YOLO
    for wid in YOLO_IDS:
        print(f"[prefetch] YOLO {wid}", flush=True)
        YOLO(wid)  # triggers the download to CWD if missing
    print("[prefetch] YOLO weights ready")


def prefetch_rfdetr() -> None:
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
    for cls in (RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge):
        print(f"[prefetch] RF-DETR {cls.__name__}", flush=True)
        cls()  # constructor fetches the COCO-pretrained base checkpoint
    print("[prefetch] RF-DETR weights ready")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "yolo"):
        prefetch_yolo()
    if which in ("all", "rfdetr"):
        prefetch_rfdetr()
    print("[prefetch] done")
