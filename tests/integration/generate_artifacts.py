"""One-off generator for the committed integration-test artifacts.

Run from the repo root (requires the ``[dev]`` extra for scikit-learn):

    .venv/bin/python tests/integration/generate_artifacts.py

Trains every black-box model exactly once with fixed seeds and writes
everything to ``tests/integration/artifacts/``. The test suite never trains a
model and never downloads a dataset; it only loads these files. The script is
deterministic and rerunnable: identical outputs given identical library
versions (see ``manifest.json``).

Standalone note: the pet case reads read-only legacy data
(``ExplanationExampleRemote/DATA``) from a checkout of the research monorepo
next to this repo. The committed artifacts keep this repo's test suite green
without it; only rerunning ``make_pets`` needs the monorepo present.

Artifacts produced:
- ``tabular_binary.npz`` + ``breast_cancer_lr.joblib`` — scaled breast-cancer
  features, LogisticRegression predictions as black-box labels.
- ``digits_tabular.npz`` + ``digits_rf.joblib`` — 500 flattened digit images,
  RandomForest predictions as black-box labels.
- ``digits_image.npz`` — binary (0 vs 1) and 10-class subsets of digit images
  shaped ``(N, 1, 8, 8)``; the 10-class labels come from a trained TinyCNN,
  the binary labels from a dense-vs-sparse TinyCNN on the same images.
- ``cnn_multiclass.pt`` / ``cnn_binary.pt`` — state_dicts of the TinyCNNs
  used as image black boxes (architecture in ``tests/integration/cnn.py``).
- ``reviews.txt`` — one fixed film-review snippet per line.
- ``manifest.json`` — library versions + per-file sha256/shape provenance.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import models as tv_models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn import TinyCNN

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
# legacy research data: read-only provenance for the pet experiment. Lives in
# the research monorepo, NOT in this repo — make_pets() refuses to run
# without it; everything else regenerates standalone.
REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_DATA = REPO_ROOT / "ExplanationExampleRemote" / "DATA"
COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)
PETS_PER_CLASS = 10


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def record(manifest, name, arrays=None):
    path = os.path.join(ARTIFACTS, name)
    entry = {"sha256": sha256(path)}
    if arrays is not None:
        entry["shapes"] = {k: list(v.shape) for k, v in arrays.items()}
    manifest["files"][name] = entry


def make_tabular_binary(manifest):
    data = load_breast_cancer()
    scaler = StandardScaler()
    x = scaler.fit_transform(data.data).astype(np.float32)
    lr = LogisticRegression(max_iter=5000, random_state=0, solver="liblinear")
    lr.fit(x, data.target)
    y = lr.predict(x).astype(np.int64)

    np.savez_compressed(os.path.join(ARTIFACTS, "tabular_binary.npz"), x=x, y=y)
    joblib.dump({"scaler": scaler, "logistic_regression": lr}, os.path.join(ARTIFACTS, "breast_cancer_lr.joblib"))
    record(manifest, "tabular_binary.npz", {"x": x, "y": y})
    record(manifest, "breast_cancer_lr.joblib")
    print(f"tabular binary: x{x.shape}, logistic-regression train accuracy {lr.score(x, data.target):.3f}")


def _digit_subset(per_class):
    digits = load_digits()
    x = digits.data / 16.0  # scale to [0, 1]
    rng = np.random.RandomState(0)
    keep = []
    for c in range(10):
        idx = np.flatnonzero(digits.target == c)
        keep.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return x[keep], digits.target[keep]


def make_digits_tabular(manifest, x, y_true):
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(x, y_true)
    y = rf.predict(x).astype(np.int64)

    np.savez_compressed(os.path.join(ARTIFACTS, "digits_tabular.npz"), x=x.astype(np.float32), y=y)
    joblib.dump({"random_forest": rf}, os.path.join(ARTIFACTS, "digits_rf.joblib"))
    record(manifest, "digits_tabular.npz", {"x": x, "y": y})
    record(manifest, "digits_rf.joblib")
    print(f"tabular multiclass: x{x.shape}, forest oob-style train accuracy {rf.score(x, y_true):.3f}")


def _train_tiny_cnn(x, y, n_classes, epochs=40, batch=64, in_channels=1):
    torch.manual_seed(0)
    cnn = TinyCNN(n_classes=n_classes, in_channels=in_channels)
    optimiser = torch.optim.AdamW(cnn.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.int64)
    for _ in range(epochs):
        cnn.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch):
            sel = perm[i : i + batch]
            optimiser.zero_grad()
            loss_fn(cnn(xt[sel]), yt[sel]).backward()
            optimiser.step()
    cnn.eval()
    return cnn


def make_digits_image(manifest, per_class_binary=60):
    # binary subset (0 vs 1), kept for robustness/future items even though the
    # current suite exercises binary images at unit level
    digits = load_digits()
    images = (digits.data / 16.0).reshape(-1, 1, 8, 8).astype(np.float32)
    rng = np.random.RandomState(1)
    bin_idx = []
    for c in (0, 1):
        idx = np.flatnonzero(digits.target == c)
        bin_idx.append(rng.choice(idx, size=min(per_class_binary, len(idx)), replace=False))
    bin_idx = np.concatenate(bin_idx)
    rng.shuffle(bin_idx)

    # multiclass subset: same selection rule as the tabular artifacts so both
    # views describe the same 500 digits
    xm, ym_true = _digit_subset(50)
    xm = xm.astype(np.float32).reshape(-1, 1, 8, 8)
    xt = torch.from_numpy(xm)

    cnn_multi = _train_tiny_cnn(xm, ym_true, n_classes=10)
    with torch.no_grad():
        y_multi = cnn_multi(xt).argmax(1).numpy().astype(np.int64)
    multi_accuracy = float((y_multi == ym_true).mean())

    # binary dense-vs-sparse black box on the same images: labels split at the
    # median ink mass, the exact signal a pixel-shared surrogate can express
    ink = xm.reshape(len(xm), -1).sum(1)
    dense_sparse = (ink >= np.median(ink)).astype(np.int64)
    cnn_binary = _train_tiny_cnn(xm, dense_sparse, n_classes=2)
    with torch.no_grad():
        y_binary = cnn_binary(xt).argmax(1).numpy().astype(np.int64)
    binary_accuracy = float((y_binary == dense_sparse).mean())

    torch.save(cnn_multi.state_dict(), os.path.join(ARTIFACTS, "cnn_multiclass.pt"))
    torch.save(cnn_binary.state_dict(), os.path.join(ARTIFACTS, "cnn_binary.pt"))

    # coordinate-augmented view: intensity plus intensity-gated normalized
    # row/col grids as extra channels. Pooling then keeps
    # {mass, sum(row*ink), sum(col*ink)} — mass and center-of-mass — giving
    # the spatially-shared surrogate real (if partial) class signal.
    rr, cc = np.mgrid[0:8, 0:8]
    x_coords = np.concatenate(
        [xm, (rr / 7.0)[None] * xm, (cc / 7.0)[None] * xm], axis=1
    ).astype(np.float32)
    cnn_coords = _train_tiny_cnn(x_coords, ym_true, n_classes=10, in_channels=3)
    with torch.no_grad():
        y_coords = cnn_coords(torch.from_numpy(x_coords)).argmax(1).numpy().astype(np.int64)
    coords_accuracy = float((y_coords == ym_true).mean())
    torch.save(cnn_coords.state_dict(), os.path.join(ARTIFACTS, "cnn_multiclass_coords.pt"))

    np.savez_compressed(
        os.path.join(ARTIFACTS, "digits_image.npz"),
        x_bin=images[bin_idx],
        y_bin=digits.target[bin_idx].astype(np.int64),
        x_multi=xm,
        y_multi=y_multi,
        y_binary=y_binary,
        x_coords=x_coords,
        y_coords=y_coords,
    )
    record(manifest, "cnn_multiclass.pt")
    record(manifest, "cnn_binary.pt")
    record(manifest, "cnn_multiclass_coords.pt")
    record(
        manifest,
        "digits_image.npz",
        {"x_bin": images[bin_idx], "x_multi": xm, "y_multi": y_multi, "y_binary": y_binary, "x_coords": x_coords},
    )
    print(f"image multiclass: x{xm.shape}, tiny-cnn accuracy {multi_accuracy:.3f}")
    print(f"image binary (dense vs sparse): tiny-cnn accuracy {binary_accuracy:.3f}")
    print(f"image multiclass (coordinate channels): tiny-cnn accuracy {coords_accuracy:.3f}")

def make_pets(manifest):
    """Miniature cat-vs-dog experiment mirroring the legacy pipeline.

    Copies a fixed subset of raw JPEGs (read-only) from the legacy research
    data together with the recorded GPT-4o-mini labels, then fits the same
    RoT-on-MobileNet-features pipeline once to commit *reference
    explanations* used as regression anchors. No feature maps are stored:
    at test time the features and explanations are recomputed afresh.

    Requires a checkout of the research monorepo next to this repo
    (``ExplanationExampleRemote/DATA``); raises a clear error otherwise.
    """
    if not LEGACY_DATA.is_dir():
        raise RuntimeError(
            f"make_pets needs the research monorepo at {LEGACY_DATA} "
            "(read-only pet data); the committed artifacts stay valid without it."
        )
    labels_df = pd.read_csv(LEGACY_DATA / "openai_classification_results.csv")
    gpt_labels = dict(zip(labels_df["filename"], labels_df["api_prediction"]))

    pet_dir = Path(ARTIFACTS) / "pet_images"
    rows = []
    for cls in ("Cat", "Dog"):
        out_dir = pet_dir / cls.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        images = sorted((LEGACY_DATA / "PetImages" / cls).glob("*.jpg"), key=lambda p: int(p.stem))
        taken = 0
        for src in images:
            rel = f"PetImages/{cls}/{src.name}"
            if rel not in gpt_labels:
                continue  # CSV covers only the first 2500 images per class
            dst = out_dir / src.name
            shutil.copyfile(src, dst)
            try:  # the Kaggle dump contains a handful of corrupt files
                Image.open(dst).convert("RGB").load()
            except Exception:  # noqa: BLE001 — any decode failure disqualifies the image
                dst.unlink()
                continue
            rows.append(
                {"filename": f"{cls.lower()}/{src.name}", "ground_truth": cls.lower(), "gpt_label": gpt_labels[rel]}
            )
            taken += 1
            if taken == PETS_PER_CLASS:
                break
        assert taken == PETS_PER_CLASS, f"only found {taken} usable {cls} images"

    pd.DataFrame(rows).to_csv(Path(ARTIFACTS) / "pets_labels.csv", index=False)

    weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    backbone = tv_models.mobilenet_v3_small(weights=weights).eval()
    transform = weights.transforms()
    batch = torch.stack([transform(Image.open(pet_dir / r["filename"]).convert("RGB")) for r in rows])
    with torch.no_grad():
        features = backbone.features(batch).numpy().astype(np.float32)

    y_gpt = np.array([1 if r["gpt_label"] == "dog" else 0 for r in rows], dtype=np.int64)
    from ruleofthumb import fit_image

    torch.manual_seed(0)
    torch.set_num_threads(1)
    exp = fit_image(y_gpt, features, epochs=300, batch_size=5000, learning_rate=0.05, seed=0, device="cpu")
    reference = exp.get_explanation(features).astype(np.float32)  # signed class-"dog" heatmaps
    rot_accuracy = float((exp.predict(torch.from_numpy(features)).cpu().numpy() == y_gpt).mean())

    np.savez_compressed(Path(ARTIFACTS) / "pet_reference_explanations.npz", heatmaps=reference)
    record(manifest, "pet_reference_explanations.npz", {"heatmaps": reference})
    record(manifest, "pets_labels.csv")
    gpt_accuracy = float((y_gpt == [r["ground_truth"] == "dog" for r in rows]).mean())
    print(f"pets: {len(rows)} images ({PETS_PER_CLASS}/class), gpt accuracy {gpt_accuracy:.2f}, rot fit accuracy {rot_accuracy:.2f}")


def _fit_tabular_models(x, y_true):
    """Fit the typical DS model trio; returns models + their predictions."""
    models = {
        "gbm": GradientBoostingClassifier(random_state=0),
        "svc": SVC(kernel="rbf", random_state=0),
        "mlp": MLPClassifier(random_state=0, max_iter=1000),
    }
    predictions = {}
    for name, model in models.items():
        model.fit(x, y_true)
        predictions[name] = model.predict(x).astype(np.int64)
    return models, predictions


def _save_tabular_case(manifest, stem, x, feature_names, models, predictions, y_true):
    y = {f"y_{name}": pred for name, pred in predictions.items()}
    np.savez_compressed(os.path.join(ARTIFACTS, f"{stem}.npz"), x=x.astype(np.float32), **y)
    joblib.dump({"feature_names": list(feature_names), "models": models}, os.path.join(ARTIFACTS, f"{stem}_models.joblib"))
    arrays = {"x": x.astype(np.float32)}
    arrays.update(y)
    record(manifest, f"{stem}.npz", arrays)
    record(manifest, f"{stem}_models.joblib")
    scores = ", ".join(f"{n} {(p == y_true).mean():.3f}" for n, p in predictions.items())
    print(f"{stem}: x{x.shape}, black-box train accuracy — {scores}")


def make_compas(manifest, n_rows=800):
    """ProPublica COMPAS two-year recidivism with the canonical filters."""
    df = pd.read_csv(COMPAS_URL)
    df = df[
        df.days_b_screening_arrest.between(-30, 30)
        & (df.is_recid != -1)
        & (df.c_charge_degree != "O")
        & (df.score_text != "N/A")
    ]
    numeric = ["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"]
    categorical = ["sex", "race", "c_charge_degree"]
    features = pd.get_dummies(df[numeric + categorical], columns=categorical, drop_first=True).astype(float)
    y_true = df["two_year_recid"].to_numpy()

    rng = np.random.RandomState(0)
    idx = rng.choice(len(features), size=n_rows, replace=False)
    scaler = StandardScaler()
    x = scaler.fit_transform(features.iloc[idx])
    names = list(features.columns)

    models, predictions = _fit_tabular_models(x, y_true[idx])
    joblib.dump({"scaler": scaler, "raw_feature_names": numeric + categorical}, os.path.join(ARTIFACTS, "compas_preprocessing.joblib"))
    _save_tabular_case(manifest, "compas", x, names, models, predictions, y_true[idx])


def make_wine(manifest):
    wine = load_wine()
    scaler = StandardScaler()
    x = scaler.fit_transform(wine.data)
    y_true = wine.target

    models, predictions = _fit_tabular_models(x, y_true)
    joblib.dump({"scaler": scaler}, os.path.join(ARTIFACTS, "wine_preprocessing.joblib"))
    _save_tabular_case(manifest, "wine", x, list(wine.feature_names), models, predictions, y_true)


REVIEWS = [
    "Brilliant acting, wonderful pacing, and a superb script with no awful scenes at all.",
    "A brilliant film with wonderful performances and a sharp, moving script.",
    "Absolutely loved it. The direction is confident and the ending is devastating in the best way.",
    "One of the best films of the year: funny, humane and beautifully shot.",
    "A charming, warm-hearted movie that earns every laugh and every tear.",
    "Superb acting and a tight, gripping story from start to finish.",
    "Delightful from the first frame to the last. I could watch it again immediately.",
    "The cast is phenomenal and the soundtrack perfectly matches the mood.",
    "A masterpiece of quiet storytelling with stunning cinematography.",
    "Clever, exciting and full of heart. Highly recommended.",
    "An instant classic. The screenplay crackles and nothing is wasted.",
    "I was bored senseless. The plot drags and the characters are paper-thin.",
    "A dull, predictable mess that wastes its talented cast.",
    "Terrible pacing, wooden dialogue and an ending that insults the audience.",
    "One of the worst films I have seen this year. Painfully unfunny.",
    "The story is incoherent and the special effects look cheap.",
    "Utterly forgettable. I checked my watch more often than I followed the plot.",
    "A clumsy, overlong slog with no likeable characters to root for.",
    "Awful writing and worse editing. Avoid at all costs.",
    "The film tries hard but collapses into a confusing, joyless muddle.",
    "Shallow, noisy and exhausting. A huge disappointment.",
    "What a gorgeous, tender piece of cinema. Every scene sings.",
    "Mediocre at best. It has a couple of good moments buried in tedium.",
    "Fresh, original and surprisingly emotional. A real gem.",
    "A riveting thriller with excellent performances across the board.",
    "Lifeless and derivative. You have seen everything here done better elsewhere.",
    "Charming and bittersweet, with a lead performance worth remembering.",
    "Crude humor and a lazy script make this a chore to sit through.",
    "Vibrant, inventive filmmaking that rewards a second viewing.",
    "The worst kind of sequel: cynical, loud and completely unnecessary.",
    "Graceful, haunting and wonderfully acted. Don't miss it.",
]


def write_reviews(manifest):
    path = os.path.join(ARTIFACTS, "reviews.txt")
    with open(path, "w") as f:
        f.write("\n".join(REVIEWS) + "\n")
    record(manifest, "reviews.txt")
    print(f"reviews: {len(REVIEWS)} fixed snippets")


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    manifest = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "transformers": __import__("transformers").__version__,
        "sklearn": __import__("sklearn").__version__,
        # pinned backbone revisions the integration tier resolves (see
        # conftest TEST_DEVICE/_deterministic and ruleofthumb.embed/vision)
        "mobilenet_weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        "modernbert_revision": __import__("ruleofthumb").DEFAULT_TEXT_REVISION,
        "sst2_revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
        "files": {},
    }
    make_tabular_binary(manifest)
    x_tab, y_tab = _digit_subset(50)
    make_digits_tabular(manifest, x_tab, y_tab)
    make_digits_image(manifest)
    write_reviews(manifest)
    make_pets(manifest)
    make_compas(manifest)
    make_wine(manifest)

    path = os.path.join(ARTIFACTS, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"manifest written to {path}")


if __name__ == "__main__":
    main()
