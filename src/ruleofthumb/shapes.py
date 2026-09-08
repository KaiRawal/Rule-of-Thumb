"""Learned elementwise response functions for non-linear additive RoT models.

A response function ``s: R -> R`` is applied elementwise to the raw inputs
before the linear RoT term: ``imp[k, i] = a[k, i] * (s(x[i]) + b[k, i])``.
Both bundled responses are *residual* (``s(x) = x + correction``) with
zero-initialised coefficients, so an unfitted non-linear model is exactly
the linear model and training grows non-linearity only where it reduces
loss. Parameters are shared across all input elements, so the parameter
budget stays independent of input size.

Select via the ``nonlinear=`` argument accepted by every RoT constructor and
factory: a string (bundled defaults) or a dict whose ``"type"`` key names
the response and whose remaining keys are forwarded as constructor
hyperparameters, e.g. ``{"type": "rbf", "n_bases": 32}``.
"""

import numpy as np
import torch
from torch import nn


class RBFResponse(nn.Module):
    """Residual Gaussian-bump response: ``s(x) = x + sum_w v_w * exp(-((x - c_w) / sigma_w)**2)``.

    Centres start evenly spaced on ``[-1, 1]`` and bandwidths at the spacing;
    both are learnable.
    """

    def __init__(self, n_bases=16):
        super().__init__()
        if n_bases < 1:
            raise ValueError("n_bases must be >= 1")
        centres = torch.linspace(-1.0, 1.0, n_bases)
        spacing = float(centres[1] - centres[0]) if n_bases > 1 else 1.0
        self.centres = nn.Parameter(centres)
        self.log_bandwidths = nn.Parameter(torch.full((n_bases,), float(np.log(spacing))))
        self.coefficients = nn.Parameter(torch.zeros(n_bases))

    def forward(self, points):
        flat = points.reshape(-1)
        scaled = (flat[:, None] - self.centres[None, :]) / self.log_bandwidths.exp()[None, :]
        bumps = (self.coefficients[None, :] * torch.exp(-(scaled**2))).sum(-1)
        return (flat + bumps).reshape(points.shape)


class HingeResponse(nn.Module):
    """Residual SELU-hinge response: ``s(x) = x + sum_w v_w * selu(x - c_w)``.

    Knots start evenly spaced on ``[-1, 1]`` and are learnable.
    """

    def __init__(self, n_bases=16):
        super().__init__()
        if n_bases < 1:
            raise ValueError("n_bases must be >= 1")
        self.knots = nn.Parameter(torch.linspace(-1.0, 1.0, n_bases))
        self.coefficients = nn.Parameter(torch.zeros(n_bases))

    def forward(self, points):
        flat = points.reshape(-1)
        hinges = (self.coefficients[None, :] * torch.nn.functional.selu(flat[:, None] - self.knots[None, :])).sum(-1)
        return (flat + hinges).reshape(points.shape)


_RESPONSES = {"rbf": RBFResponse, "hinge": HingeResponse}


def resolve_nonlinear(spec):
    """Normalise a ``nonlinear=`` spec to ``(type_name, hyperparameters)``.

    Accepts a string (bundled defaults) or a dict with a ``"type"`` key whose
    remaining entries become constructor hyperparameters.
    """
    if isinstance(spec, str):
        type_name, kwargs = spec, {}
    elif isinstance(spec, dict):
        kwargs = dict(spec)
        type_name = kwargs.pop("type", None)
    else:
        raise TypeError(f"nonlinear must be a string or a dict with a 'type' key, got {spec!r}")
    if type_name not in _RESPONSES:
        raise ValueError(f"unknown nonlinear type {type_name!r}; expected one of {sorted(_RESPONSES)}")
    return type_name, kwargs


def build_response(spec, device=None):
    """Instantiate the response module named by a ``nonlinear=`` spec."""
    type_name, kwargs = resolve_nonlinear(spec)
    module = _RESPONSES[type_name](**kwargs)
    if device is not None:
        module = module.to(device)
    return module


def normalised_spec(spec):
    """Reduce a ``nonlinear=`` spec to a primitives-only dict for persistence."""
    type_name, kwargs = resolve_nonlinear(spec)
    return {"type": type_name, **kwargs}
