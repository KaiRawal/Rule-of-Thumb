"""High-level Rule-of-Thumb explainer facade.

One entry point for all three modalities:

- :func:`fit` — auto-detects the modality from the input shape;
- :func:`fit_tabular` / :func:`fit_text` / :func:`fit_image` — explicit,
  each accepting only its own keyword arguments.

All factories return a fitted :class:`Explainer` whose ``get_explanation``
returns signed, SHAP-comparable importances: class-1 contributions for
binary tasks, full per-class output for ``n_classes > 2``.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import numpy as np
import torch

from ruleofthumb.core import RoT
from ruleofthumb.embed import embed_texts
from ruleofthumb.image import RoTImage, load_images
from ruleofthumb.text import RoTText

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"})


def _is_string_batch(x):
    """True when ``x`` is a non-empty list/tuple of raw strings."""
    return isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(s, str) for s in x)


def _is_path_batch(x):
    """True when ``x`` is a non-empty list/tuple of image file paths."""
    return _is_string_batch(x) and all(os.path.splitext(s)[1].lower() in _IMAGE_EXTENSIONS for s in x)


def _as_float_inputs(x_inputs):
    return torch.from_numpy(np.asarray(x_inputs)).to(torch.float)


def _as_labels(y_outputs):
    return torch.as_tensor(np.asarray(y_outputs)).flatten()


def _check_label_range(labels, n_classes):
    """Fail clearly when labels fall outside ``[0, n_classes)`` instead of leaking a torch indexing error."""
    if labels.numel() == 0:
        raise ValueError("y_outputs must be non-empty")
    lo, hi = int(labels.min()), int(labels.max())
    if lo < 0 or hi >= n_classes:
        bad = lo if lo < 0 else hi
        raise ValueError(
            f"labels contain class {bad} but n_classes={n_classes}; pass the matching n_classes "
            "or remap the labels (e.g. binary-vs-rest) before fitting"
        )
    return labels


class Explainer:
    """Fitted Rule-of-Thumb explainer.

    Create instances through :func:`fit`, :func:`fit_tabular`,
    :func:`fit_text` or :func:`fit_image` rather than constructing directly.

    A text explainer fitted from raw strings, or an image explainer fitted
    from file paths, remembers how to load its inputs: every public method
    then accepts the same strings / paths in place of numeric arrays,
    re-loading (and re-deriving padding masks) on each call.

    The underlying model is available as :attr:`model` (a :class:`~ruleofthumb.core.RoT`
    subclass) for full manual control; the reveal-pipeline methods
    (:meth:`get_order`, :meth:`ordered_predict`, :meth:`score_ordering`, ...)
    delegate to it verbatim.
    """

    def __init__(self, model, modality, string_embedder=None, image_loader=None):
        self._model = model
        self._modality = modality
        self._string_embedder = string_embedder
        self._image_loader = image_loader

    @property
    def modality(self):
        """Which pipeline this explainer runs: ``"tabular"``, ``"text"`` or ``"image"``."""
        return self._modality

    @property
    def model(self):
        """The fitted underlying RoT model."""
        return self._model

    def _resolve_native(self, x):
        """Load a raw string/path batch; returns ``(points tensor, mask tensor)`` or ``None``."""
        if not _is_string_batch(x):
            return None
        if self._modality == "text":
            if self._string_embedder is None:
                raise ValueError(
                    "this explainer was fitted on embedding arrays and cannot consume raw strings; "
                    "pass arrays or refit from strings"
                )
            embedded = self._string_embedder(list(x))
            return torch.from_numpy(embedded.embeddings).to(self._model.device), torch.from_numpy(embedded.attention_mask)
        if self._image_loader is None:
            raise ValueError(
                "this explainer was fitted on image arrays and cannot consume file paths; "
                "pass arrays or refit from paths"
            )
        batch = self._image_loader(list(x))
        return torch.from_numpy(batch.images).to(self._model.device), torch.from_numpy(batch.mask)

    def get_explanation(self, x_numpy, *, mask=None) -> np.ndarray:
        """Return signed importances, comparable to SHAP values.

        For binary tasks (``n_classes == 2``) the result holds the class-1
        ("positive class") contributions: positive values are evidence toward
        class 1, negative toward class 0. For ``n_classes > 2`` the full
        per-class importances are returned; the class axis is never collapsed.

        Shapes: tabular ``(N, d)`` / ``(N, K, d)``; text ``(N, tokens)`` /
        ``(N, K, tokens)`` (embedding dims summed per token); image
        ``(N, H, W)`` / ``(N, K, H, W)`` (channels summed per pixel). Padded
        positions receive exactly zero importance when a mask is supplied.

        Padding arguments depend on the modality: tabular takes none, text
        and image accept ``mask`` (or none at all for raw strings / file
        paths — padding is derived automatically). Build text masks from
        per-sample token counts with
        :func:`ruleofthumb.text.lengths_to_mask`.
        """
        resolved = self._resolve_native(x_numpy)
        if resolved is not None:
            if mask is not None:
                raise ValueError("raw inputs derive their padding automatically; pass no mask")
            x, native_mask = resolved
        else:
            x = torch.from_numpy(np.asarray(x_numpy)).to(self._model.device)
            native_mask = None
        if self._modality == "tabular":
            if mask is not None:
                raise ValueError("tabular explanations take no mask")
            imp = self._model.importance(x)
        elif self._modality == "text":
            if resolved is None:
                native_mask = None if mask is None else torch.as_tensor(np.asarray(mask)).to(torch.bool)
            imp = self._model.importance(x, mask=native_mask)
        else:
            if resolved is not None:
                imp = self._model.importance(x, mask=native_mask)
            else:
                if mask is not None:
                    mask = torch.as_tensor(np.asarray(mask)).to(torch.bool)
                imp = self._model.importance(x, mask=mask)
        imp = self._model._reduce_to_units(imp).detach().cpu().numpy()
        if self._model.classes == 2:
            imp = imp[:, 1]
        return imp

    _MASK_METHODS = frozenset({"get_order", "score", "predict"})

    def _delegate(self, name, args, kwargs):
        """Call the raw-model method, loading a leading string/path batch for text/image."""
        if args and _is_string_batch(args[0]) and self._modality in ("text", "image"):
            if kwargs.get("mask") is not None:
                raise ValueError("raw inputs derive their padding automatically; pass mask=None")
            points, mask = self._resolve_native(args[0])
            args = (points,) + args[1:]
            if name in self._MASK_METHODS:
                kwargs["mask"] = mask
        return getattr(self._model, name)(*args, **kwargs)

    def get_order(self, points, *, mask=None, granularity="unit"):
        """Rank reveal units by absolute importance, most important first.

        See :meth:`ruleofthumb.core.RoT.get_order`. Text array inputs take a
        boolean ``mask`` of shape ``(N, tokens)``; padded positions are ranked
        last as ``-1``. Raw strings / file paths derive their padding
        automatically (pass no mask). ``score_ordering`` takes no mask: the
        ``-1`` entries in the returned order already encode the padding.
        """
        if self._modality == "tabular" and mask is not None:
            raise ValueError("tabular explainers take no mask")
        if self._modality == "text" and mask is not None and not _is_string_batch(points):
            mask = torch.as_tensor(np.asarray(mask)).to(torch.bool)
        return self._delegate("get_order", (points,), {"mask": mask, "granularity": granularity})

    def ordered_predict(self, *args, **kwargs):
        """See :meth:`ruleofthumb.core.RoT.ordered_predict`. Accepts raw strings / file paths."""
        return self._delegate("ordered_predict", args, kwargs)

    def score(self, *args, **kwargs):
        """See :meth:`ruleofthumb.core.RoT.score`. Accepts raw strings / file paths."""
        return self._delegate("score", args, kwargs)

    def predict(self, *args, **kwargs):
        """See :meth:`ruleofthumb.core.RoT.predict`. Accepts raw strings / file paths."""
        return self._delegate("predict", args, kwargs)

    def score_ordering(self, *args, **kwargs):
        """See :meth:`ruleofthumb.core.RoT.score_ordering`. Accepts raw strings / file paths."""
        return self._delegate("score_ordering", args, kwargs)

    def save(self, path):
        """Persist this explainer to ``path`` for reloading without refitting.

        Saves the underlying RoT model's weights and configuration; use
        :func:`load_explainer` to restore it. Native string / file-path
        ingestion is not persisted — a reloaded explainer consumes numeric
        arrays.
        """
        model = self._model
        config = {
            "classes": int(model.classes),
            "sample_shape": [int(s) for s in model.sample_shape],
            "dropout_rate": float(model.dropout_rate),
            "use_BCE_loss": bool(model.use_BCE_loss),
        }
        if self._modality == "text":
            config["l1_penalty"] = float(model.l1_penalty)
        if getattr(model, "nonlinear_spec", None) is not None:
            config["nonlinear"] = dict(model.nonlinear_spec)
        payload = {
            "ruleofthumb_format": _PERSISTENCE_FORMAT,
            "modality": self._modality,
            "config": config,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "mins": _to_payload_value(model.mins),
            "maxs": _to_payload_value(model.maxs),
        }
        torch.save(payload, path)


_PERSISTENCE_FORMAT = 1


def _to_payload_value(value):
    """Reduce ``mins`` / ``maxs`` to weights_only-safe primitives."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value if isinstance(value, float) else np.asarray(value).tolist()


def _from_payload_value(value, device):
    if isinstance(value, list):
        return torch.tensor(value, device=device)
    return value


def load_explainer(path: str | os.PathLike, *, device: Any | None = None) -> Explainer:
    """Load an :class:`Explainer` saved with :meth:`Explainer.save`.

    Args:
        path: file written by :meth:`Explainer.save`.
        device: optional torch device for the restored model; ``None``
            auto-detects cuda > mps > cpu.

    Returns:
        A fitted :class:`Explainer` consuming numeric arrays (native
        string / file-path ingestion is not persisted).
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("ruleofthumb_format") != _PERSISTENCE_FORMAT:
        raise ValueError(f"{path!r} is not a ruleofthumb explainer file")
    modality = payload["modality"]
    config = dict(payload["config"])
    config["sample_shape"] = tuple(config["sample_shape"])
    classes = {"tabular": RoT, "text": RoTText, "image": RoTImage}[modality]
    model = classes(device=device, **config)
    model.load_state_dict(payload["state_dict"])
    model.mins = _from_payload_value(payload["mins"], model.device)
    model.maxs = _from_payload_value(payload["maxs"], model.device)
    return Explainer(model, modality)


def fit(y_outputs, x_inputs, *, modality="auto", **kwargs):
    """Fit an :class:`Explainer` to black-box outputs, auto-detecting the modality.

    ``modality="auto"`` dispatches on the input: a list of image file paths →
    image, any other list of raw strings → text; otherwise on
    ``x_inputs.ndim``: 2 → tabular, 3 → text ``(N, tokens, embedding)``,
    4 → image ``(N, C, H, W)``. Pass ``modality="tabular" | "text" | "image"``
    explicitly to override. The remaining keyword arguments are forwarded to
    the matching factory (:func:`fit_tabular`, :func:`fit_text` or
    :func:`fit_image`).
    """
    if modality == "auto":
        if _is_path_batch(x_inputs):
            modality = "image"
        elif _is_string_batch(x_inputs):
            modality = "text"
        else:
            ndim = np.asarray(x_inputs).ndim
            modality = {2: "tabular", 3: "text", 4: "image"}.get(ndim)
            if modality is None:
                raise ValueError(f"cannot infer a modality from input ndim={ndim}; pass modality= explicitly")
    factories = {"tabular": fit_tabular, "text": fit_text, "image": fit_image}
    if modality not in factories:
        raise ValueError(f"unknown modality: {modality!r}")
    return factories[modality](y_outputs, x_inputs, **kwargs)


def fit_tabular(
    y_outputs,
    x_inputs,
    *,
    epochs=500,
    batch_size=5000,
    learning_rate=0.05,
    dropout_rate=0.5,
    pretrain_epochs=5,
    weight_decay=0.01,
    seed=None,
    n_classes=2,
    device=None,
    nonlinear=None,
):
    """Fit a tabular :class:`Explainer` on ``(N, d)`` feature inputs."""
    labels = _check_label_range(_as_labels(y_outputs), n_classes)
    model = RoT(n_classes, (x_inputs.shape[1],), dropout_rate=dropout_rate, device=device, nonlinear=nonlinear)
    model.fit(
        _as_float_inputs(x_inputs),
        labels,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        pretrain_epochs=pretrain_epochs,
        weight_decay=weight_decay,
        seed=seed,
    )
    return Explainer(model, "tabular")


def fit_text(
    y_outputs,
    x_inputs,
    *,
    mask=None,
    tokenizer=None,
    model=None,
    l1_penalty=0.01,
    epochs=500,
    batch_size=5000,
    learning_rate=0.05,
    dropout_rate=0.5,
    pretrain_epochs=5,
    weight_decay=0.01,
    seed=None,
    n_classes=2,
    device=None,
    nonlinear=None,
):
    """Fit a text :class:`Explainer` on ``(N, tokens, embedding)`` inputs or raw strings.

    Pass a list of raw strings and they are embedded automatically with the
    bundled default HuggingFace model (:data:`ruleofthumb.DEFAULT_TEXT_MODEL`);
    supply ``tokenizer`` / ``model`` (a pre-loaded ``AutoModel``) to override
    it. Padding masks are derived automatically — ``mask`` must not be given
    alongside strings.

    For array inputs, padding is explicit: pass ``mask`` (an ``(N, T)``
    boolean array marking real tokens). HuggingFace tokenizer
    ``attention_mask`` tensors and :func:`ruleofthumb.text.lengths_to_mask`
    output compose directly. Without a mask, every token is treated as real
    data.

    Explainers fitted from strings accept the same strings back in every
    public method; each call re-embeds the texts.
    """
    string_embedder = None
    if _is_string_batch(x_inputs):
        if mask is not None:
            raise ValueError("raw strings derive their padding automatically; pass no mask")
        string_embedder = functools.partial(embed_texts, tokenizer=tokenizer, model=model, device=device)
        embedded = string_embedder(list(x_inputs))
        x_inputs = embedded.embeddings
        mask = torch.from_numpy(embedded.attention_mask)
    elif mask is not None:
        mask = torch.as_tensor(np.asarray(mask)).to(torch.bool)
    rot = RoTText(
        n_classes,
        (x_inputs.shape[1], x_inputs.shape[2]),
        dropout_rate=dropout_rate,
        l1_penalty=l1_penalty,
        device=device,
        nonlinear=nonlinear,
    )
    rot.fit(
        _as_float_inputs(x_inputs),
        _check_label_range(_as_labels(y_outputs), n_classes),
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        mask=mask,
        pretrain_epochs=pretrain_epochs,
        weight_decay=weight_decay,
        seed=seed,
    )
    return Explainer(rot, "text", string_embedder=string_embedder)


def fit_image(
    y_outputs,
    x_inputs,
    *,
    mask=None,
    size=None,
    transform=None,
    epochs=500,
    batch_size=5000,
    learning_rate=0.05,
    dropout_rate=0.5,
    pretrain_epochs=5,
    weight_decay=0.01,
    seed=None,
    n_classes=2,
    device=None,
    nonlinear=None,
):
    """Fit an image :class:`Explainer` on ``(N, C, H, W)`` inputs or image file paths.

    Pass a list of image file paths (PNG / JPEG / ...) and they are decoded
    automatically (RGB, ``[0, 1]`` floats): with ``size=(height, width)``
    every image is resized and centre-cropped to that common size; without
    it, native sizes are kept and smaller images are zero-padded. Validity
    masks are derived automatically — ``mask=`` must not be given alongside
    paths. Supply ``transform=`` (a PIL Image -> tensor callable) to replace
    the default pipeline entirely, e.g. a torchvision weights transform.

    For array inputs, pass a boolean validity ``mask`` of shape ``(N, H, W)``
    alongside padded batches (see :func:`ruleofthumb.image.pad_images`);
    masked-out pixels receive exactly zero importance.

    Explainers fitted from paths accept the same paths back in every public
    method; each call re-loads the files.
    """
    image_loader = None
    if _is_string_batch(x_inputs):
        if mask is not None:
            raise ValueError("raw image files derive their padding automatically; pass mask=None")
        image_loader = functools.partial(load_images, size=size, transform=transform)
        loaded = image_loader(list(x_inputs))
        x_inputs = loaded.images
        mask = torch.from_numpy(loaded.mask)
    elif mask is not None:
        mask = torch.as_tensor(np.asarray(mask)).to(torch.bool)
    rot = RoTImage(n_classes, (x_inputs.shape[1],), dropout_rate=dropout_rate, device=device, nonlinear=nonlinear)
    rot.fit(
        _as_float_inputs(x_inputs),
        _check_label_range(_as_labels(y_outputs), n_classes),
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        mask=mask,
        pretrain_epochs=pretrain_epochs,
        weight_decay=weight_decay,
        seed=seed,
    )
    return Explainer(rot, "image", image_loader=image_loader)
