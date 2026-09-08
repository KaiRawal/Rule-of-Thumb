"""Integration tests: fitted explainers survive save/load on real artifacts.

Each case fits on committed artifacts, saves to a temporary file, reloads
with :func:`ruleofthumb.load_explainer` and asserts identical explanations
and reveal-curve outputs. The subprocess case proves survival across a real
process boundary.
"""

import subprocess
import sys

import numpy as np
import torch

from ruleofthumb import fit_image, fit_tabular, fit_text, load_explainer

SEED = 0


def test_tabular_round_trip(tmp_path, tabular_binary):
    x, y = tabular_binary["x"], tabular_binary["y"]
    exp = fit_tabular(y, x, epochs=200, batch_size=500, learning_rate=0.05, seed=SEED)
    path = tmp_path / "tabular.rotx"
    exp.save(str(path))

    loaded = load_explainer(str(path))
    assert np.allclose(exp.get_explanation(x), loaded.get_explanation(x))
    xt, yt = torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))
    order_a, order_b = exp.get_order(xt), loaded.get_order(xt)
    assert np.array_equal(order_a, order_b)
    assert np.allclose(
        exp.score_ordering(xt, yt, order_a).numpy(),
        loaded.score_ordering(xt, yt, order_b).numpy(),
    )


def test_text_round_trip(tmp_path, text_sst2):
    x, mask, y = text_sst2["embeddings"], text_sst2["attention_mask"].numpy(), text_sst2["y"]
    exp = fit_text(y, x, attention_mask=mask, epochs=200, batch_size=500, learning_rate=0.05, seed=SEED)
    path = tmp_path / "text.rotx"
    exp.save(str(path))

    loaded = load_explainer(str(path))
    assert np.allclose(exp.get_explanation(x, attention_mask=mask), loaded.get_explanation(x, attention_mask=mask))
    xt = torch.from_numpy(x)
    mt = torch.from_numpy(mask)
    assert np.array_equal(exp.get_order(xt, mask=mt), loaded.get_order(xt, mask=mt))


def test_image_round_trip(tmp_path, image_multiclass):
    x, y = image_multiclass["x"], image_multiclass["y_binary"]
    exp = fit_image(y, x, epochs=300, batch_size=5000, learning_rate=0.05, seed=SEED)
    path = tmp_path / "image.rotx"
    exp.save(str(path))

    loaded = load_explainer(str(path))
    assert np.allclose(exp.get_explanation(x), loaded.get_explanation(x))
    assert np.allclose(exp.predict(torch.from_numpy(x)).cpu().numpy(), loaded.predict(torch.from_numpy(x)).cpu().numpy())


def test_subprocess_load_survives_process_boundary(tmp_path, tabular_binary):
    """A fresh interpreter loads the saved explainer and reproduces predictions."""
    x, y = tabular_binary["x"], tabular_binary["y"]
    exp = fit_tabular(y, x, epochs=200, batch_size=500, learning_rate=0.05, seed=SEED)
    model_path = tmp_path / "explainer.rotx"
    exp.save(str(model_path))

    data_path, preds_path = tmp_path / "x.npy", tmp_path / "preds.npy"
    np.save(data_path, x)
    code = "\n".join(
        [
            "import numpy as np",
            "from ruleofthumb import load_explainer",
            f"exp = load_explainer({str(model_path)!r})",
            f"x = np.load({str(data_path)!r})",
            f"np.save({str(preds_path)!r}, exp.predict(x).cpu().numpy())",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)

    parent_preds = exp.predict(torch.from_numpy(x)).cpu().numpy()
    assert (np.load(preds_path) == parent_preds).all()
