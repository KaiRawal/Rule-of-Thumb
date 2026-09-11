"""Rule of Thumb: explaining AI systems using partial information.

Experimental 0.0.x pre-alpha: entirely vibe-coded from hand-written
research code. APIs may break without notice; verify explanations
before trusting them.
"""

from ruleofthumb.core import RoT
from ruleofthumb.embed import DEFAULT_TEXT_MODEL, DEFAULT_TEXT_REVISION, TextEmbeddings, embed_texts
from ruleofthumb.explain import Explainer, fit, fit_image, fit_tabular, fit_text, load_explainer
from ruleofthumb.image import ImageBatch, load_images, pad_images
from ruleofthumb.text import lengths_to_mask, pad_sequences
from ruleofthumb.tune import AutotuneResult, autotune
from ruleofthumb.vision import DEFAULT_IMAGE_MODEL, ImageEmbeddings, embed_images

__version__ = "0.0.2"

__all__ = [
    "DEFAULT_IMAGE_MODEL",
    "DEFAULT_TEXT_MODEL",
    "DEFAULT_TEXT_REVISION",
    "AutotuneResult",
    "Explainer",
    "ImageBatch",
    "ImageEmbeddings",
    "RoT",
    "TextEmbeddings",
    "__version__",
    "autotune",
    "embed_images",
    "embed_texts",
    "fit",
    "fit_image",
    "fit_tabular",
    "fit_text",
    "lengths_to_mask",
    "load_explainer",
    "load_images",
    "pad_images",
    "pad_sequences",
]
