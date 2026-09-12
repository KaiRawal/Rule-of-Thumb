"""MIT1003 integration: RoT-image saliency vs human fixation boxes.

Needs the external cache (``fetch_external.py``); skips without it.
Committed artifacts hold the RoT weights (maps arm: full 500-set fit;
pixel arm: 150-subset fit), the black-box predictions, the eval indices
and the IG/occlusion reference maps for a fixed 20-subset.

Box protocol: top-10%-mass threshold, 8-connected components (min area 1%
of the image) on both the saliency and the fixation-density maps, then
greedy-IoU matching, pointing-game hits and mass coverage — identical
machinery for every method and baseline.
"""

import json
import os

import numpy as np
import pytest
import torch
from _external import rebuild_image_explainer, require_cache
from PIL import Image

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

pytest.importorskip("torchvision")


def _extract_boxes(sal, mass_q=10, min_frac=0.01):
    from scipy.ndimage import label

    thr = np.quantile(sal.ravel(), 1 - mass_q / 100)
    lab, n = label((sal >= thr).astype(np.int32), structure=np.ones((3, 3), int))
    boxes = []
    for c in range(1, n + 1):
        ys, xs = np.nonzero(lab == c)
        if len(ys) / sal.size >= min_frac:
            boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    return boxes


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0 + 1) * max(0, iy1 - iy0 + 1)
    aa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    bb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / (aa + bb - inter)


def _pointing(sal, gt_boxes):
    peak = np.unravel_index(np.argmax(sal), sal.shape)
    return any(g[0] <= peak[1] <= g[2] and g[1] <= peak[0] <= g[3] for g in gt_boxes)


def _up(arr, size=128):
    return np.array(Image.fromarray(arr.astype(np.float32)).resize((size, size), Image.BILINEAR))


def _pred_class_abs_maps(explainer, Xn, classes):
    m = explainer.model
    a = m.a.detach()
    b = m.b.detach()
    out = []
    with torch.no_grad():
        for i, k in enumerate(classes):
            x = torch.as_tensor(Xn[i : i + 1])
            r = m._respond(x)
            impk = a[int(k)][:, None, None] * (r[0] + b[int(k)][:, None, None])
            out.append(impk.abs().sum(0).numpy())
    return np.array(out)


@pytest.fixture(scope="module")
def sal():
    X = np.load(require_cache("sal_X.npy"))
    F = np.load(require_cache("sal_F.npy"))
    P = np.load(os.path.join(ARTIFACTS, "sal_P.npy")).astype(np.int64)
    idx = np.load(os.path.join(ARTIFACTS, "sal_sub_idx.npy"))
    mob, _ = rebuild_image_explainer(ARTIFACTS, "sal_rot_mob")
    pix, _ = rebuild_image_explainer(ARTIFACTS, "sal_rot_pix")

    pytest.importorskip("torchvision")
    from torchvision import models as tv_models

    weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    backbone = tv_models.mobilenet_v3_small(weights=weights).eval()
    t = torch.from_numpy(X[idx].transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
    with torch.no_grad():
        Fm = torch.cat([backbone.features(t[i : i + 32]) for i in range(0, len(idx), 32)]).numpy().astype(np.float32)
    Pm = np.asarray(mob.predict(Fm, sample_chunk=25))
    mob_sals = np.array([_up(m) for m in _pred_class_abs_maps(mob, Fm, P[idx])])

    Xp = np.stack([np.array(Image.fromarray(X[j]).resize((32, 32))).transpose(2, 0, 1) for j in idx]).astype(
        np.float32
    ) / 255.0
    Pp = np.asarray(pix.predict(Xp, sample_chunk=25))
    pix_sals = np.array([_up(m) for m in _pred_class_abs_maps(pix, Xp, P[idx])])
    return {"X": X, "F": F, "P": P, "idx": idx, "Pm": Pm, "Pp": Pp,
            "mob_sals": mob_sals, "pix_sals": pix_sals}


def test_sal_mob_fidelity(sal):
    assert float(np.mean(sal["Pm"] == sal["P"][sal["idx"]])) >= 0.90


def test_sal_pixel_collapse_documented(sal):
    assert float(np.mean(sal["Pp"] == sal["P"][sal["idx"]])) <= 0.30


def _pointing_table(sal):
    fixes = sal["F"][sal["idx"]]
    yy, xx = np.mgrid[0:128, 0:128]
    center = (((xx - 64) ** 2 + (yy - 64) ** 2) <= 32**2).astype(float)
    rng = np.random.RandomState(0)
    table = {}
    for name, arr in (("mob", sal["mob_sals"]), ("pix", sal["pix_sals"]),
                      ("center", [center] * len(sal["idx"])),
                      ("rand", [rng.random((128, 128)) for _ in sal["idx"]])):
        table[name] = float(np.mean([_pointing(s, _extract_boxes(f)) for s, f in zip(arr, fixes)]))
    return table


def test_sal_pointing_order(sal):
    table = _pointing_table(sal)
    assert table["mob"] >= 0.28
    assert table["mob"] > table["pix"]
    assert table["rand"] <= 0.15


def test_sal_box_iou_order(sal):
    fixes = sal["F"][sal["idx"]]
    ious = {}
    for name, arr in (("mob", sal["mob_sals"]), ("pix", sal["pix_sals"])):
        vals = []
        for s, f in zip(arr, fixes):
            gb = _extract_boxes(f)
            pb = _extract_boxes(s)
            if gb and pb:
                vals.append(float(np.mean([max(_iou(p, g) for g in gb) for p in pb])))
        ious[name] = float(np.mean(vals))
    assert ious["mob"] > ious["pix"]


def test_sal_ig_occ_refs(sal):
    bundle = np.load(os.path.join(ARTIFACTS, "sal_base20.npz"))
    with open(os.path.join(ARTIFACTS, "sal_refs.json")) as f:
        refs = json.load(f)
    assert bundle["ig"].shape == bundle["occ"].shape == (20, 128, 128)
    assert bundle["ig"].dtype == np.uint8
    ig = bundle["ig"].astype(float) / 255 * (refs["ig_scale"][1] - refs["ig_scale"][0]) + refs["ig_scale"][0]
    occ = bundle["occ"].astype(float) / 255 * (refs["occ_scale"][1] - refs["occ_scale"][0]) + refs["occ_scale"][0]
    with open(os.path.join(ARTIFACTS, "sal_ig_idx.json")) as f:
        gi = json.load(f)
    fixes = sal["F"][gi]
    assert float(np.mean([_pointing(s, _extract_boxes(f)) for s, f in zip(ig, fixes)])) >= 0.20
    assert float(np.mean([_pointing(s, _extract_boxes(f)) for s, f in zip(occ, fixes)])) >= 0.15


def test_sal_renders(sal):
    import matplotlib

    matplotlib.use("Agg")
    from ruleofthumb import plot

    fig = plot.saliency(sal["mob_sals"][0] * np.sign(sal["mob_sals"][0] - sal["mob_sals"][0].mean()),
                        image=sal["X"][sal["idx"][0]])
    assert fig is not None
    try:
        plot.saliency(np.zeros((2, 3, 4, 4)))
    except ValueError as e:
        assert "class index" in str(e)
    else:
        raise AssertionError("unsliced multiclass saliency should raise")
