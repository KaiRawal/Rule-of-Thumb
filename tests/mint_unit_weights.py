"""Mint the committed mock RoT weights for unit tests (ToDo item 22).

Usage (repo root, venv only)::

    .venv/bin/python tests/mint_unit_weights.py [--check]

Runs the exact fits the unit tests previously performed live (CPU, seeded,
single-threaded) once, and writes raw state dicts — deliberately not
``.rotx`` — plus rebuild specs to ``tests/fixtures/``:

- ``plot_tabular[.pt|_spec.json]`` / ``plot_tabular_multi`` — plot rendering
- ``persist_{tabular,text,image}`` — persistence round-trips (fit incidental)
- ``calib_binary`` / ``calib_multi`` — calibrated mechanism recovery
- ``faith_{deletion,pointing,noise}`` — faithfulness probes
- ``ring_{linear,rbf,hinge}`` — nonlinear ring separation

Dataset builders are single-sourced from ``tests/_mock_weights.py``: the
mint fits and the tests consume identical inputs by construction.

``--check`` reloads every fixture through ``load_mock``, recomputes the key
metric each converted test asserts, and verifies it against the tightened
margin recorded in the spec (exits nonzero on mismatch).
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _mock_weights as mw
import numpy as np
import torch

FIXTURES = mw.FIXTURES


def _seed_everything():
    seed = int(os.environ.get("ROT_TEST_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _cpu_floats(value):
    tensor = torch.as_tensor(value() if callable(value) else value, dtype=torch.float32)
    return tensor.cpu().tolist()


def _save(name, explainer, checks):
    os.makedirs(FIXTURES, exist_ok=True)
    torch.save(explainer.model.state_dict(), os.path.join(FIXTURES, f"{name}.pt"))
    spec = {
        "model": explainer.modality,
        "n_classes": int(explainer.model.classes),
        "sample_shape": [int(s) for s in explainer.model.sample_shape],
        "mins": _cpu_floats(explainer.model.mins),
        "maxs": _cpu_floats(explainer.model.maxs),
        "checks": checks,
    }
    if explainer.model.nonlinear_spec is not None:
        spec["nonlinear"] = explainer.model.nonlinear_spec
    with open(os.path.join(FIXTURES, f"{name}_spec.json"), "w") as f:
        json.dump(spec, f, indent=2, sort_keys=True)
    print(f"minted {name}.pt")


def mint():
    from ruleofthumb import fit_image, fit_tabular, fit_text

    _seed_everything()

    x, _, y = mw.data_plot_tabular()
    _save(
        "plot_tabular",
        fit_tabular(y, x, epochs=30, batch_size=40, learning_rate=0.05, seed=0, device="cpu"),
        {},
    )

    x, _, y = mw.data_plot_tabular_multi()
    _save(
        "plot_tabular_multi",
        fit_tabular(y, x, epochs=8, batch_size=48, learning_rate=0.05, seed=0, n_classes=3, device="cpu"),
        {},
    )

    for modality in ("tabular", "text", "image"):
        x, mask, y = mw.data_persist(modality)
        kwargs = {"epochs": 8, "batch_size": 16, "learning_rate": 0.05, "seed": 0, "device": "cpu"}
        fit = {"tabular": fit_tabular, "text": fit_text, "image": fit_image}[modality]
        if mask is None:
            exp = fit(y, x, **kwargs)
        else:
            exp = fit(y, x, mask=mask, **kwargs)
        _save(f"persist_{modality}", exp, {})

    from test_calibrated import CALIBRATED_W, _ExactLinear

    x, _, _ = mw.data_calib_binary()
    with torch.no_grad():
        y = (_ExactLinear(CALIBRATED_W)(torch.from_numpy(x)) > 0).numpy().astype(np.int64)
    _save(
        "calib_binary",
        fit_tabular(y, x, epochs=200, batch_size=64, learning_rate=0.05, seed=0, device="cpu"),
        {"spearman": 0.9, "top3": [0, 1, 2], "argmin": 5, "sign_corr": 0.9},
    )

    x, _, _ = mw.data_calib_multi()
    y = np.argmax(x @ mw.CALIB_MULTI_W.T, axis=1).astype(np.int64)
    _save(
        "calib_multi",
        fit_tabular(y, x, epochs=200, batch_size=64, learning_rate=0.05, seed=0, n_classes=3, device="cpu"),
        {},
    )

    x, _, y = mw.data_faith_deletion()
    _save(
        "faith_deletion",
        fit_tabular(y, x, epochs=60, batch_size=64, learning_rate=0.05, seed=0, device="cpu"),
        {"deletion_gap": 0.25},
    )

    x, _, y = mw.data_faith_pointing()
    _save(
        "faith_pointing",
        fit_image(y, x, epochs=60, batch_size=8, learning_rate=0.05, seed=0, device="cpu"),
        {"hit_rate": 0.75},
    )

    x, _, y = mw.data_faith_noise()
    _save(
        "faith_noise",
        fit_tabular(y, x, epochs=100, batch_size=64, learning_rate=0.05, seed=0, device="cpu"),
        {},
    )

    x, _, y = mw.data_ring()
    _save(
        "ring_linear",
        fit_tabular(y, x, epochs=250, batch_size=500, learning_rate=0.05, seed=0, device="cpu"),
        {"max_accuracy": 0.8},
    )
    for kind in ("rbf", "hinge"):
        _save(
            f"ring_{kind}",
            fit_tabular(
                y, x, epochs=400, batch_size=500, learning_rate=0.05, seed=0, device="cpu", nonlinear=kind
            ),
            {"min_accuracy": 0.9, "min_margin_over_linear": 0.2},
        )


def _accuracy(explainer, x, y, mask=None):
    import torch as _torch

    preds = np.asarray(explainer.predict(_torch.from_numpy(x)).cpu())
    return float((preds == y).mean())


def check():
    from test_calibrated import CALIBRATED_W, _ExactLinear, _spearman

    failures = []
    for name in mw.mock_names():
        exp = mw.load_mock(name)
        with open(os.path.join(FIXTURES, f"{name}_spec.json")) as f:
            checks = json.load(f)["checks"]
        if name == "calib_binary":
            from _mock_weights import data_calib_binary

            x, _, _ = data_calib_binary()
            with torch.no_grad():
                y = (_ExactLinear(CALIBRATED_W)(torch.from_numpy(x)) > 0).numpy().astype(np.int64)
            imp = exp.get_explanation(x)
            scores = np.abs(imp).mean(0)
            truth = torch.abs(CALIBRATED_W).numpy()
            got = {
                "spearman": _spearman(scores, truth),
                "sign_corr": min(
                    float(np.corrcoef(imp[:, d], x[:, d])[0, 1]) * np.sign(CALIBRATED_W[d].item())
                    for d in range(5)
                ),
            }
            ok = (
                got["spearman"] >= checks["spearman"]
                and set(np.argsort(-scores)[:3]) == set(checks["top3"])
                and int(np.argmin(scores)) == checks["argmin"]
                and got["sign_corr"] >= checks["sign_corr"]
            )
        elif name.startswith("ring_"):
            from _mock_weights import data_ring

            x, _, y = data_ring()
            acc = _accuracy(exp, x, y)
            if name == "ring_linear":
                ok, got = acc < checks["max_accuracy"], {"accuracy": acc}
            else:
                lin = mw.load_mock("ring_linear")
                lin_acc = _accuracy(lin, x, y)
                ok, got = (
                    acc >= checks["min_accuracy"] and acc >= lin_acc + checks["min_margin_over_linear"],
                    {"accuracy": acc, "linear_accuracy": lin_acc},
                )
        elif name == "faith_deletion":
            from _mock_weights import data_faith_deletion

            x, _, y = data_faith_deletion()
            full = _accuracy(exp, x, y)
            top2 = np.argsort(-np.abs(exp.get_explanation(x)).mean(0))[:2]
            knocked = x.copy()
            knocked[:, top2] = 0.0
            gap = full - _accuracy(exp, knocked, y)
            ok, got = set(top2.tolist()) == {0, 1} and gap >= checks["deletion_gap"], {
                "gap": gap,
                "top2": top2.tolist(),
            }
        elif name == "faith_pointing":
            from _mock_weights import data_faith_pointing

            x, _, y = data_faith_pointing()
            imp = exp.get_explanation(x)
            hits = sum(
                4 <= int(np.unravel_index(int(np.argmax(np.abs(imp[i]))), (16, 16))[0]) < 8
                and 4 <= int(np.unravel_index(int(np.argmax(np.abs(imp[i]))), (16, 16))[1]) < 8
                for i in np.flatnonzero(y == 1)
            )
            rate = hits / max(1, int((y == 1).sum()))
            ok, got = rate >= checks["hit_rate"], {"hit_rate": rate}
        else:
            got = {"loaded": True}
            ok = True
        print(f"{name}: {'OK' if ok else 'FAIL'} {got}")
        if not ok:
            failures.append(name)
    if failures:
        raise SystemExit(f"mock-weight check failed for: {failures}")
    print(f"OK: all {len(mw.mock_names())} fixtures reproduce their checks")


def main():
    if "--check" in sys.argv[1:]:
        check()
    else:
        mint()


if __name__ == "__main__":
    main()
