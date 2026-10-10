"""Objects365 as a memory-mapped label cache.

Objects365 train is 1.74M images and 22.9M kept boxes; its annotation JSON is
5.5 GB (14 GB with the SAM masks inline) and expands to tens of GB of Python
objects, which `load_coco` would then hand, pickled, to every training rank.
So the labels are parsed once into compact numpy files and every process
memory-maps the same files: the OS keeps one physical copy in RAM, shared
across ranks and across forked dataloader workers.

Boxes live in the box cache. Masks come from the SAM per-image RLE tree,
decoded once to polygons into a second, memory-mapped polygon cache, so
training workers never run the RLE decode and contour trace per sample.

    python -m mayaku.data.o365 build \\
        --ann /data/o365/annotations/zhiyuan_objv2_train.json \\
        --img-root /data/o365/train \\
        --sam-root /data/o365/annotations/sam_mask_json/sam_obj365_train_1742k \\
        --out /data/o365/cache/train
    python -m mayaku.data.o365 build-polys --cache /data/o365/cache/train

The box cache is a directory of:
    boxes.npy     (N_ann, 5) float32  [dense_class, x1, y1, x2, y2] pixels
    ann_ids.npy   (N_ann,)   int64    original annotation id (joins SAM masks)
    img_off.npy   (N_img+1,) int64    boxes[img_off[i]:img_off[i+1]] is image i
    img_ids.npy   (N_img,)   int64
    img_hw.npy    (N_img, 2) int32    (height, width)
    paths.npy     (N_img+1,) int64    offsets into paths.bin
    paths.bin     utf-8 image file names, "images/v1/patchK/xxx.jpg"
    meta.json     nc, cat_ids, counts, roots
and the polygon cache adds poly_xy.npy (points), poly_poff.npy (part ->
point offsets) and poly_ann_off.npy (annotation -> part offsets). Crowd and
sub-2px boxes are dropped at build, as `load_coco` drops them.

`o365_train_labels` and `o365_val_labels` return what `CocoDetection(coco=)`
takes, so the dataset, the trainer and the evaluator need no O365 path.
"""

import argparse
import concurrent.futures as cf
import json
import os

import numpy as np

from mayaku.data.coco import keep_ann, load_coco
from mayaku.data.polygons import Polys, outline
from mayaku.data.serialize import SerializedList

# ---------------------------------------------------------------- building


def build_cache(ann, img_root, out, sam_root=None):
    """Stream the Objects365 JSON once and write the box cache."""
    import ijson
    os.makedirs(out, exist_ok=True)

    imgs = {}          # image_id -> index
    ids, hw, names = [], [], []
    with open(ann, "rb") as f:
        for im in ijson.items(f, "images.item"):
            imgs[im["id"]] = len(ids)
            ids.append(im["id"])
            hw.append((im["height"], im["width"]))
            names.append(im["file_name"])
    n_img = len(ids)
    print("images: %d" % n_img, flush=True)

    cats = set()
    rows = []          # (img_index, cat_id, x1, y1, x2, y2, ann_id)
    with open(ann, "rb") as f:
        for a in ijson.items(f, "annotations.item"):
            if not keep_ann(a):
                continue
            x, y, w, h = (float(v) for v in a["bbox"])
            cats.add(a["category_id"])
            rows.append((imgs[a["image_id"]], a["category_id"], x, y, x + w, y + h, a["id"]))
    cat_ids = sorted(cats)
    print("annotations kept: %d, categories: %d" % (len(rows), len(cat_ids)), flush=True)

    r = np.asarray(rows, dtype=np.float64)
    del rows                                             # free the ~7 GB list before the sort
    r = r[np.argsort(r[:, 0], kind="stable")]            # group by image
    boxes = np.empty((len(r), 5), np.float32)
    # cat_ids is sorted, so an id's dense index is its position in it
    boxes[:, 0] = np.searchsorted(cat_ids, r[:, 1])
    boxes[:, 1:5] = r[:, 2:6]
    ann_ids = r[:, 6].astype(np.int64)
    counts = np.bincount(r[:, 0].astype(np.int64), minlength=n_img)
    img_off = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)

    enc = [s.encode("utf-8") for s in names]
    poff = np.zeros(n_img + 1, np.int64)
    poff[1:] = np.cumsum([len(b) for b in enc])

    np.save(out + "/boxes.npy", boxes)
    np.save(out + "/ann_ids.npy", ann_ids)
    np.save(out + "/img_off.npy", img_off)
    np.save(out + "/img_ids.npy", np.asarray(ids, np.int64))
    np.save(out + "/img_hw.npy", np.asarray(hw, np.int32))
    np.save(out + "/paths.npy", poff)
    with open(out + "/paths.bin", "wb") as f:
        f.write(b"".join(enc))
    with open(out + "/meta.json", "w") as f:
        json.dump({"nc": len(cat_ids), "cat_ids": cat_ids, "n_img": n_img,
                   "n_ann": len(r), "img_root": img_root, "sam_root": sam_root}, f)
    print("wrote cache to %s" % out, flush=True)


def _sam_path(sam_root, rel):
    """SAM per-image JSON path from an image's relative file name."""
    base = os.path.basename(rel)
    patch = rel.split("/")[-2]                        # images/vX/patchK/xxx.jpg
    return os.path.join(sam_root, patch, base.rsplit(".", 1)[0] + ".json")


def _sam_segs(sam_root, rel):
    """One image's SAM masks, {annotation id (str): RLE}; empty when absent."""
    p = _sam_path(sam_root, rel)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f).get("segmentations", {})


def _poly_shard(args):
    """Decode the SAM masks for one image range to polygons, the way the
    COCO path does (`Polys.from_coco`: continuous pixel-edge outlines).
    Returns (xy, part_lens, ann_nparts): all polygon points stacked, the
    point count of each part, and the part count of each annotation
    (0 = no mask), with annotations in box-cache order."""
    cache, sam_root, lo, hi = args
    store = _Store(cache)
    xy, part_lens, ann_nparts = [], [], []
    for i in range(lo, hi):
        segs = _sam_segs(sam_root, store.rel(i))
        p = Polys.from_coco([segs.get(str(int(a)))
                             for a in store.ann_ids[store.off[i]:store.off[i + 1]]])
        xy.append(p.xy)
        part_lens.append(np.diff(p.part_off))
        ann_nparts.append(np.diff(p.inst_off))
    return np.concatenate(xy), np.concatenate(part_lens), np.concatenate(ann_nparts)


def build_poly_cache(cache, sam_root=None, workers=16, chunk=4000):
    """Decode every image's SAM masks to polygons once and store them as a
    memory-mapped CSR beside the box cache (the RLE decode and contour trace
    were ~75% of the dataloader). `sam_root` defaults to the cache's own."""
    import time
    with open(cache + "/meta.json") as f:
        m = json.load(f)
    sam_root = sam_root or m["sam_root"]
    n_img, n_ann = m["n_img"], m["n_ann"]
    tasks = [(cache, sam_root, lo, min(lo + chunk, n_img)) for lo in range(0, n_img, chunk)]
    part_lens, ann_nparts = [], []
    xy_path = cache + "/poly_xy.npy"
    t0, done, pts = time.time(), 0, 0
    header = {"descr": "<f4", "fortran_order": False, "shape": (0, 2)}
    # points go to a growable .npy: header first, shard blocks appended
    with open(xy_path, "wb") as fx:
        np.lib.format.write_array_header_2_0(fx, header)
        header_end = fx.tell()
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for xy, pl, an in ex.map(_poly_shard, tasks):
                fx.write(np.ascontiguousarray(xy, np.float32).tobytes())
                part_lens.append(pl)
                ann_nparts.append(an)
                pts += len(xy)
                done += 1
                if done % 20 == 0:
                    print("  polys: %d/%d chunks, %d points, %.0fs"
                          % (done, len(tasks), pts, time.time() - t0), flush=True)
    # the row count is known now: rewrite the header in place
    with open(xy_path, "r+b") as fx:
        np.lib.format.write_array_header_2_0(fx, {**header, "shape": (pts, 2)})
        assert fx.tell() == header_end, "the .npy header changed length"
    part_lens, ann_nparts = np.concatenate(part_lens), np.concatenate(ann_nparts)
    assert len(ann_nparts) == n_ann, (len(ann_nparts), n_ann)
    poff = np.zeros(len(part_lens) + 1, np.int64)
    poff[1:] = np.cumsum(part_lens)
    ann_off = np.zeros(n_ann + 1, np.int64)
    ann_off[1:] = np.cumsum(ann_nparts)
    np.save(cache + "/poly_poff.npy", poff)
    np.save(cache + "/poly_ann_off.npy", ann_off)
    # the coordinates are pixel-edge outlines; caches built before say nothing
    # and hold raw contour indices, which `_polys` outlines on read
    with open(cache + "/meta.json", "w") as f:
        json.dump({**m, "poly_coords": "edge"}, f)
    print("poly cache: %d points, %d parts over %d annotations, %.0fs"
          % (pts, len(part_lens), n_ann, time.time() - t0), flush=True)


# ----------------------------------------------------------------- reading


class _Store:
    """The cache's memory maps. Pickles as its path and reopens on load, so
    a spawned rank or worker maps the same files rather than receiving a
    copy of their contents (an `np.memmap` pickles as its whole array)."""

    def __init__(self, cache, masks=False, limit=None):
        self.cache, self.masks, self.limit = cache, masks, limit
        self._open()

    @property
    def root(self):
        """The directory the cached file names are relative to."""
        return self.meta["img_root"]

    def _open(self):
        with open(os.path.join(self.cache, "meta.json")) as f:
            self.meta = json.load(f)
        def mm(name):
            return np.load(os.path.join(self.cache, name), mmap_mode="r")

        self.off, self.ann_ids = mm("img_off.npy"), mm("ann_ids.npy")
        self.boxes, self.ids, self.hw = mm("boxes.npy"), mm("img_ids.npy"), mm("img_hw.npy")
        self.blob = np.memmap(os.path.join(self.cache, "paths.bin"), dtype=np.uint8, mode="r")
        self.poff = mm("paths.npy")
        self.poly = None
        if self.masks:
            if not os.path.exists(os.path.join(self.cache, "poly_ann_off.npy")):
                raise FileNotFoundError(f"{self.cache} has no polygon cache; run build-polys")
            self.poly = (mm("poly_xy.npy"), mm("poly_poff.npy"), mm("poly_ann_off.npy"))

    def __getstate__(self):
        return {"cache": self.cache, "masks": self.masks, "limit": self.limit}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def __len__(self):
        n = len(self.off) - 1
        return n if self.limit is None else min(n, self.limit)

    def rel(self, i):
        return bytes(self.blob[self.poff[i]:self.poff[i + 1]]).decode("utf-8")


class _Column:
    """One per-image field of the cache as a read-only sequence; `get` is a
    module-level function, so the column pickles."""

    def __init__(self, store, get):
        self.store, self.get = store, get

    def __len__(self):
        return len(self.store)

    def __getitem__(self, i):
        return self.get(self.store, int(i))


def _id(s, i):
    return int(s.ids[i])


def _file(s, i):
    return os.path.join(s.root, s.rel(i))


def _labels(s, i):
    return np.array(s.boxes[s.off[i]:s.off[i + 1]], np.float32)      # a writable copy


def _polys(s, i):
    xy, poff, aoff = s.poly
    a0, a1 = s.off[i], s.off[i + 1]                   # the image's annotations
    p0, p1 = int(aoff[a0]), int(aoff[a1])             # their parts
    part_off = (np.asarray(poff[p0:p1 + 1]) - poff[p0]).astype(np.int32)
    pts = xy[int(poff[p0]):int(poff[p1])]
    # a cache from before `poly_coords` holds raw findContours indices
    pts = np.array(pts, np.float32) if s.meta.get("poly_coords") == "edge" else outline(pts, part_off)
    inst = np.asarray(aoff[a0:a1 + 1])
    return Polys(pts, part_off, (inst - inst[0]).astype(np.int32), np.diff(inst) > 0)


class O365Labels:
    """The training labels from a cache, field for field what `load_coco`
    returns (`mayaku.data.coco.CocoLabels`), backed by memory maps.

    `class_names` defaults to the category ids as strings; pass the val
    file's names so checkpoints carry real ones. `limit` keeps the first
    `limit` images only (smoke runs). There is no annotation file
    behind it: training never scores against the training split.
    """

    def __init__(self, cache, masks=False, class_names=None, limit=None):
        self.store = _Store(cache, masks, limit)
        meta = self.store.meta
        self.ann_path = self.kpt_ann_path = cache
        self.num_degenerate = 0
        self.cat_ids = list(meta["cat_ids"])
        if class_names is not None and len(class_names) != len(self.cat_ids):
            raise ValueError(f"{len(class_names)} class names for {len(self.cat_ids)} classes")
        self.class_names = list(class_names) if class_names else [str(c) for c in self.cat_ids]
        self.kpt_names, self.kpt_flip_pairs = [], ()
        self.ids = _Column(self.store, _id)
        self.files = _Column(self.store, _file)
        self.labels = _Column(self.store, _labels)
        self.polys = _Column(self.store, _polys) if masks else []
        self.kpts = []

    @property
    def shapes(self):
        return self.store.hw[:len(self.store)]


def o365_train_labels(cache, masks=False, class_names=None, limit=None):
    """The training labels in `cache` (see `O365Labels`)."""
    return O365Labels(cache, masks=masks, class_names=class_names, limit=limit)


def o365_val_labels(json_path, images_root, masks=False):
    """The Objects365 merged val file (boxes + inline RLE masks, small enough
    to parse) as `CocoLabels`. Its file names are `patchK/objects365_vN_x.jpg`
    while the images sit under `<root>/vN/patchK/`; the version is in the
    base name, so it is spliced back in."""
    coco = load_coco(str(images_root), str(json_path), masks=masks)
    root = str(images_root)

    def where(path):
        rel = os.path.relpath(path, root)
        return os.path.join(root, "v2" if "_v2_" in os.path.basename(rel) else "v1", rel)

    coco.files = SerializedList([where(coco.files[i]) for i in range(len(coco.files))])
    return coco


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="the box cache")
    b.add_argument("--ann", required=True)
    b.add_argument("--img-root", required=True, help="dir the file_names are relative to")
    b.add_argument("--sam-root", default=None, help="SAM per-image mask tree (patchK/*.json)")
    b.add_argument("--out", required=True)
    p = sub.add_parser("build-polys", help="the polygon cache beside a box cache")
    p.add_argument("--cache", required=True)
    p.add_argument("--sam-root", default=None)
    p.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build_cache(a.ann, a.img_root, a.out, a.sam_root)
    else:
        build_poly_cache(a.cache, a.sam_root, workers=a.workers)


if __name__ == "__main__":
    main()
