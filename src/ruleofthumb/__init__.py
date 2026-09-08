"""Rule of Thumb: explaining AI systems using partial information.

Experimental 0.0.x pre-alpha: entirely vibe-coded from hand-written
research code. APIs may break without notice; verify explanations
before trusting them.
"""

from ruleofthumb.core import RoT
from ruleofthumb.embed import DEFAULT_TEXT_MODEL, TextEmbeddings, embed_texts
from ruleofthumb.explain import Explainer, fit, fit_image, fit_tabular, fit_text, load_explainer
from ruleofthumb.image import ImageBatch, load_images, pad_images
from ruleofthumb.text import pad_sequences, sentinel_mask
from ruleofthumb.tune import AutotuneResult, autotune

__version__ = "0.0.1"

__all__ = [
    "DEFAULT_TEXT_MODEL",
    "AutotuneResult",
    "Explainer",
    "ImageBatch",
    "RoT",
    "TextEmbeddings",
    "__version__",
    "autotune",
    "embed_texts",
    "fit",
    "fit_image",
    "fit_tabular",
    "fit_text",
    "load_explainer",
    "load_images",
    "pad_images",
    "pad_sequences",
    "sentinel_mask",
]
