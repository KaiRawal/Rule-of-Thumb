"""Fetch external human-annotation datasets into a gitignored cache.

Populates ``tests/integration/external_cache/`` (never committed; CI runs
this as a setup step, ``pytest`` itself trains nothing and downloads
nothing). Developers without network copy the listed files from a machine
that has them — the script verifies sizes/shapes either way.

Datasets:
- HateXPlain posts + rationales (GitHub raw, punyajoy/HateXplain):
  ``dataset.json`` (~12MB) + ``post_id_divisions.json``.
- MIT1003 photographs + fixation maps (official MIT WherePeopleLook):
  ``ALLSTIMULI.zip`` (~225MB) + ``ALLFIXATIONMAPS.zip`` (~9MB).

Derived cache contents (all deterministic, seeds fixed below):
- ``hx_full.json``: full train/val/test texts, human labels, test rationales,
  target tags (majority vote, ties dropped, mean rationale masks).
- ``hx_box.joblib``: TF-IDF(8000) + logistic-regression black box trained on
  human train labels; ``hx_ybb_{train,val,test}.npy`` its predictions.
- ``sal_X.npy`` (500,128,128,3) uint8 photographs, ``sal_F.npy``
  (500,128,128) float32 fixation densities, ``sal_names.json``.
- ``sal_sub_idx.npy``: seed-7 150-subset; ``sal_P.npy``: stock ImageNet
  ResNet18 top-1 predictions (``x/255`` preprocessing, no normalisation);
  ``sal_mob.npy``: MobileNetV3-Small features @128px for the 500-set.

Usage: ``.venv/bin/python tests/integration/fetch_external.py [--only hx|sal]``
"""

import json
import os
import sys
import urllib.request
import zipfile
from collections import Counter

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_cache")

HX_BASE = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/"
HX_FILES = {"dataset.json": 12256170, "post_id_divisions.json": 591921}
MIT_BASE = "http://people.csail.mit.edu/tjudd/WherePeopleLook/"
MIT_FILES = {"ALLSTIMULI.zip": 225 * 1024 * 1024, "ALLFIXATIONMAPS.zip": 9 * 1024 * 1024}

LAB = {"hatespeech": 1, "offensive": 1, "normal": 0}


def _get(url, dest, expect):
    if os.path.exists(dest) and os.path.getsize(dest) >= 0.9 * expect:
        print(f"cached {dest}")
        return
    print(f"downloading {url} ...", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"saved {dest} ({os.path.getsize(dest)} bytes)", flush=True)


def fetch_hx():
    import joblib
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    os.makedirs(CACHE, exist_ok=True)
    for name, size in HX_FILES.items():
        _get(HX_BASE + name, os.path.join(CACHE, name), size)
    with open(os.path.join(CACHE, "dataset.json")) as f:
        ds = json.load(f)
    with open(os.path.join(CACHE, "post_id_divisions.json")) as f:
        div = json.load(f)

    def parse(pid):
        v = ds[pid]
        toks = v["post_tokens"]
        votes = [LAB[a["label"]] for a in v["annotators"]]
        c = Counter(votes)
        if c[0] == c[1]:
            return None
        y = 1 if c[1] > c[0] else 0
        fixed = []
        for r in v.get("rationales") or []:
            r = list(r)[: len(toks)] + [0] * max(0, len(toks) - len(r))
            fixed.append(r)
        rat = np.mean(fixed, axis=0) if fixed else np.zeros(len(toks))
        tgts = [t for a in v["annotators"] for t in a.get("target", [])]
        tgt = Counter(tgts).most_common(1)[0][0] if tgts else "None"
        return " ".join(toks), y, np.asarray(rat, dtype=float), tgt

    out = {}
    for split in ("train", "val", "test"):
        T, Y, R, G, drop = [], [], [], [], 0
        for pid in div[split]:
            r = parse(pid)
            if r is None:
                drop += 1
                continue
            t, y, rat, tgt = r
            T.append(t)
            Y.append(y)
            R.append(rat)
            G.append(tgt)
        out[split] = (T, np.array(Y), R, G)
        print(f"hx {split}: n={len(T)} dropped={drop}", flush=True)

    vec = TfidfVectorizer(max_features=8000, token_pattern=r"(?u)\b\w+\b")
    Xtr = vec.fit_transform(out["train"][0])
    bb = LogisticRegression(max_iter=500).fit(Xtr, out["train"][1])
    joblib.dump({"vec": vec, "bb": bb}, os.path.join(CACHE, "hx_box.joblib"))
    for split in ("train", "val", "test"):
        T, Y, _, _ = out[split]
        pred = bb.predict(vec.transform(T))
        print(f"hx {split}: box-vs-human={np.mean(pred == Y):.4f}", flush=True)
        np.save(os.path.join(CACHE, f"hx_ybb_{split}.npy"), pred)
    with open(os.path.join(CACHE, "hx_full.json"), "w") as f:
        json.dump(
            {
                "train": {"texts": out["train"][0], "y": out["train"][1].tolist()},
                "val": {"texts": out["val"][0], "y": out["val"][1].tolist()},
                "test": {
                    "texts": out["test"][0],
                    "y": out["test"][1].tolist(),
                    "rat": [r.tolist() for r in out["test"][2]],
                    "tgt": out["test"][3],
                },
            },
            f,
        )
    print("hx cache complete", flush=True)


def fetch_sal():
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights, mobilenet_v3_small, resnet18

    os.makedirs(CACHE, exist_ok=True)
    for name, size in MIT_FILES.items():
        _get(MIT_BASE + name, os.path.join(CACHE, name), size)
    for name in MIT_FILES:
        with zipfile.ZipFile(os.path.join(CACHE, name)) as z:
            z.extractall(os.path.join(CACHE, "mit1003"))
    base = os.path.join(CACHE, "mit1003")
    stim = os.path.join(base, "ALLSTIMULI")
    fixd = os.path.join(base, "ALLFIXATIONMAPS")
    if not (os.path.isdir(stim) and os.path.isdir(fixd)):
        stim, fixd = None, None
        for root, _, files in os.walk(base):
            if "MACOSX" in root:
                continue
            if stim is None and any(f.lower().endswith(".jpeg") for f in files):
                stim = root
            if fixd is None and any("_fixmap" in f.lower() for f in files):
                fixd = root
        assert stim is not None and fixd is not None, "stimulus/fixation dirs not found"
    pairs = []
    for f in sorted(os.listdir(stim)):
        stem = os.path.splitext(f)[0]
        cand = stem + "_fixMap.jpg"
        if f.lower().endswith((".jpeg", ".jpg")) and os.path.exists(os.path.join(fixd, cand)):
            pairs.append((os.path.join(stim, f), os.path.join(fixd, cand), stem))
    rng = np.random.RandomState(0)
    sel = rng.choice(len(pairs), 500, replace=False)
    Xs, Fs, names = [], [], []
    for i in sel:
        p, q, stem = pairs[i]
        Xs.append(np.array(Image.open(p).convert("RGB").resize((128, 128))))
        f = np.array(Image.open(q).convert("L").resize((128, 128))).astype(np.float64)
        Fs.append(f / f.sum())
        names.append(stem)
    X = np.stack(Xs)
    F = np.stack(Fs).astype(np.float32)
    np.save(os.path.join(CACHE, "sal_X.npy"), X)
    np.save(os.path.join(CACHE, "sal_F.npy"), F)
    with open(os.path.join(CACHE, "sal_names.json"), "w") as f:
        json.dump(names, f)
    print(f"sal 500-set: X {X.shape} F {F.shape}", flush=True)

    bb = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    with torch.no_grad():
        xb = torch.from_numpy(X.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
        P = torch.cat([bb(xb[i : i + 64]).argmax(1) for i in range(0, 500, 64)]).numpy()
    np.save(os.path.join(CACHE, "sal_P.npy"), P)
    print(f"sal BB spread: {len(np.unique(P))} classes", flush=True)

    mob = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    feats = mob.features.eval()
    with torch.no_grad():
        t = torch.nn.functional.interpolate(xb, size=128, mode="bilinear")
        Fm = torch.cat([feats(t[i : i + 32]) for i in range(0, 500, 32)]).numpy().astype(np.float32)
    np.save(os.path.join(CACHE, "sal_mob.npy"), Fm)
    print(f"sal mob feats: {Fm.shape}", flush=True)

    idx = np.random.RandomState(7).choice(500, 150, replace=False)
    np.save(os.path.join(CACHE, "sal_sub_idx.npy"), idx)
    print("sal cache complete", flush=True)


if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else "hx,sal"
    if "hx" in only:
        fetch_hx()
    if "sal" in only:
        fetch_sal()
