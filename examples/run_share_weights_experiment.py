import argparse
import inspect
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import models as tv

from ruleofthumb import fit_image
from ruleofthumb import plot as rot_plot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(REPO, "tests", "integration", "artifacts")
DEFAULT_OUT = os.path.join(ARTIFACTS, "share_weights_results")

SEED = 0
EPOCHS = 300
BATCH = 5000
LR = 0.05
N_SALIENCY = 8
EXPECTED_N = 1797

ALL_BACKBONES = ["raw", "coords", "mobilenet_v3_small", "resnet18", "convnext_tiny", "efficientnet_b0"]

SPECS = {
    "mobilenet_v3_small": (
        tv.mobilenet_v3_small,
        tv.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        lambda m, b: m.features(b),
    ),
    "convnext_tiny": (tv.convnext_tiny, tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, lambda m, b: m.features(b)),
    "efficientnet_b0": (tv.efficientnet_b0, tv.EfficientNet_B0_Weights.IMAGENET1K_V1, lambda m, b: m.features(b)),
    "resnet18": (
        tv.resnet18,
        tv.ResNet18_Weights.IMAGENET1K_V1,
        lambda m, b: torch.nn.Sequential(*list(m.children())[:-2])(b),
    ),
}


def preflight():
    if "share_weights" not in inspect.signature(fit_image).parameters:
        sys.exit(
            "ERROR: this ruleofthumb build has no share_weights= flag.\n"
            "  Check out the branch that adds it:  git checkout todo-16\n"
            "  then reinstall:                     uv pip install -e '.[image,plot,dev]'"
        )

    path = os.path.join(ARTIFACTS, "digits_image.npz")
    if not os.path.exists(path):
        sys.exit(f"ERROR: missing {path} (run tests/integration/generate_artifacts.py)")
    n = len(np.load(path)["x_multi"])
    if n != EXPECTED_N:
        print(
            f"WARNING: digits_image.npz holds {n} images, the published table used {EXPECTED_N}.\n"
            f"         Upstream main ships the older 500-image artifact; the fork's commit\n"
            f"         761f894 replaced it. Your numbers will NOT match the table.\n",
            file=sys.stderr,
        )
    return n


def resolve_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tv_features(x_gray, builder, weights, extract, device):
    backbone = builder(weights=weights).eval().to(device)
    transform = weights.transforms()
    out = []
    with torch.no_grad():
        for start in range(0, len(x_gray), 256):
            chunk = x_gray[start : start + 256]
            batch = torch.stack(
                [
                    transform(
                        Image.fromarray((chunk[i, 0] * 255).clip(0, 255).astype(np.uint8), mode="L").convert("RGB")
                    )
                    for i in range(len(chunk))
                ]
            ).to(device)
            out.append(extract(backbone, batch).cpu().numpy().astype(np.float32))
    del backbone
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out, 0)


def build_inputs(name, data, device):
    x_gray = data["x_multi"].astype(np.float32)
    y = data["y_multi"].astype(np.int64)

    if name == "raw":
        return x_gray, y, "raw pixels (C=1)"
    if name == "coords":
        return data["x_coords"].astype(np.float32), data["y_coords"].astype(np.int64), "coordinate channels (C=3)"
    if name not in SPECS:
        sys.exit(f"ERROR: unknown backbone {name!r}; choose from {ALL_BACKBONES}")

    builder, weights, extract = SPECS[name]
    return _tv_features(x_gray, builder, weights, extract, device), y, f"{name} frozen trunk"


def split(y):
    rng = np.random.RandomState(SEED)
    train, test = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        cut = round(0.8 * len(idx))
        train.append(idx[:cut])
        test.append(idx[cut:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(test))


def save_saliency(exp, x_test, y_test, digits_gray, test_idx, out_dir, mode):
    os.makedirs(out_dir, exist_ok=True)
    sel = np.arange(min(N_SALIENCY, len(x_test)))
    imp = exp.get_explanation(x_test[sel])
    pred = exp.predict(x_test[sel]).numpy()
    for j in sel:
        heat = imp[j, pred[j]] if imp.ndim == 4 else imp[j]
        heat_t = torch.from_numpy(np.asarray(heat, dtype=np.float32))[None, None]
        heat_big = torch.nn.functional.interpolate(heat_t, size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
        digit_t = torch.from_numpy(digits_gray[test_idx[j], 0].astype(np.float32))[None, None]
        digit_big = torch.nn.functional.interpolate(digit_t, size=(64, 64), mode="nearest")[0, 0].numpy()
        rgb = np.repeat(digit_big[:, :, None], 3, axis=2)
        rgb = (rgb - rgb.min()) / max(float(np.ptp(rgb)), 1e-8)
        ax = rot_plot.saliency(heat_big.numpy(), image=rgb, power=0.7)
        fig = ax.get_figure() if hasattr(ax, "get_figure") else plt.gcf()
        fig.suptitle(f"{mode} | true={y_test[j]} pred={pred[j]}", fontsize=9)
        fig.savefig(
            os.path.join(out_dir, f"{mode}_{j:02d}_true{y_test[j]}_pred{pred[j]}.png"), dpi=110, bbox_inches="tight"
        )
        plt.close(fig)


def run_backbone(name, out_root, device):
    data = np.load(os.path.join(ARTIFACTS, "digits_image.npz"))
    digits_gray = data["x_multi"].astype(np.float32)
    t0 = time.time()
    x, y, label = build_inputs(name, data, device)
    feat_secs = time.time() - t0

    train_idx, test_idx = split(y)
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "backbone": name,
        "label": label,
        "device": device,
        "feature_shape": list(x.shape),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "majority_baseline": float(np.bincount(y).max() / len(y)),
        "feature_extraction_secs": round(feat_secs, 1),
        "modes": {},
    }

    for share in (True, False):
        mode = "shared" if share else "unshared"
        t1 = time.time()
        exp = fit_image(
            y_train,
            x_train,
            n_classes=10,
            share_weights=share,
            epochs=EPOCHS,
            batch_size=BATCH,
            learning_rate=LR,
            seed=SEED,
            device=device,
        )
        m = exp.model
        params = int(m.a.numel() + m.b.numel() + m.g.numel())
        train_acc = float((exp.predict(x_train).numpy() == y_train).mean())
        test_acc = float((exp.predict(x_test).numpy() == y_test).mean())
        result["modes"][mode] = {
            "a_shape": list(m.a.shape),
            "parameters": params,
            "train_fidelity": round(train_acc, 4),
            "test_fidelity": round(test_acc, 4),
            "generalisation_gap": round(train_acc - test_acc, 4),
            "fit_secs": round(time.time() - t1, 1),
        }
        save_saliency(exp, x_test, y_test, digits_gray, test_idx, os.path.join(out_dir, "saliency"), mode)
        np.save(os.path.join(out_dir, f"importances_{mode}.npy"), exp.get_explanation(x_test[:N_SALIENCY]))
        print(f"  [{name}/{mode}] train={train_acc:.4f} test={test_acc:.4f} params={params:,}", flush=True)

    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def write_summary(results, out_root):
    first = results[0]
    lines = [
        "# Experiment 1 - Digits, 10-class shared vs unshared RoT weights",
        "",
        (
            f"N_train={first['n_train']}  N_test={first['n_test']}  "
            f"majority={first['majority_baseline']:.3f}  stratified 80/20, seed 0"
        ),
        "",
        "| backbone | features | sh-train | sh-test | sh-params | un-train | un-test | un-params | test delta |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        s, u = r["modes"]["shared"], r["modes"]["unshared"]
        feat = "x".join(str(v) for v in r["feature_shape"][1:])
        lines.append(
            f"| {r['backbone']} | {feat} | {s['train_fidelity']:.4f} | {s['test_fidelity']:.4f} | "
            f"{s['parameters']:,} | {u['train_fidelity']:.4f} | {u['test_fidelity']:.4f} | "
            f"{u['parameters']:,} | {u['test_fidelity'] - s['test_fidelity']:+.4f} |"
        )
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_root, "SUMMARY.md"), "w") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=ALL_BACKBONES, help=f"subset of {ALL_BACKBONES}")
    ap.add_argument("--device", default="auto", help="auto (default) | cuda | mps | cpu")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    preflight()
    device = resolve_device(args.device)
    heavy = [b for b in args.backbones if b not in ("raw", "coords")]
    print(f"device={device}  backbones={args.backbones}  out={args.out}")
    if heavy and device == "cpu":
        print(f"NOTE: {len(heavy)} backbone(s) on CPU will take a while; raw/coords alone run in ~20s.")

    torch.manual_seed(SEED)
    os.makedirs(args.out, exist_ok=True)
    results = [run_backbone(name, args.out, device) for name in args.backbones]
    print()
    print(write_summary(results, args.out))


if __name__ == "__main__":
    main()
