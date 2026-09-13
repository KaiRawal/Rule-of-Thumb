"""Generate committed docs figures (light + dark pairs) from dummy data.

Only numpy + matplotlib + ruleofthumb.plot (no training, no downloads,
no shap). Deterministic; CPU seconds.

Usage: PYTHONPATH=src .venv/bin/python docs-src/_figures/generate_figures.py
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "_static", "figures")
DPI = 150
DARK_BG = "#1c1e24"
RED, BLUE = "#d62728", "#1f77b4"


def _save(name, fig):
    fig.savefig(os.path.join(OUT, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _pair(name, draw):
    os.makedirs(OUT, exist_ok=True)
    with plt.style.context("default"):
        _save(f"{name}.light.png", draw(dark=False))
    with plt.style.context("dark_background"):
        _save(f"{name}.dark.png", draw(dark=True))


def _hero(dark=False):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    if dark:
        fig.patch.set_facecolor(DARK_BG)
    steps = [
        ("Unopenable\nmodel", "y = model(X)\nanswers only"),
        ("Simple\nstand-in", "copies the\nanswers"),
        ("What\nmattered", "+0.8  -0.3\n+0.5  +0.1"),
        ("Same answer\nfewer inputs", "reveal best\nfirst \u2197"),
    ]
    for ax, (title, body) in zip(axes, steps):
        ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(0.5, 0.30, body, ha="center", va="center", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig.suptitle("Rule of Thumb: which inputs drove this answer?", fontsize=13)
    fig.tight_layout()
    return fig


def _card_tabular(dark=False):
    fig, ax = plt.subplots(figsize=(4, 2.6))
    names = ["size", "age", "zip", "color"]
    vals = [0.8, -0.45, 0.25, 0.05]
    colors = [RED if v > 0 else BLUE for v in vals]
    ax.barh(names, vals, color=colors)
    ax.set_title("Tabular: one number per column")
    ax.set_xlabel("toward \u2192 red / \u2190 blue against")
    fig.tight_layout()
    return fig


def _card_text(dark=False):
    from ruleofthumb import plot as rot_plot

    fig = rot_plot.text_matplotlib(
        np.array([-0.1, 0.9, 0.2]), ["a", "wonderful", "film"], width=6
    )
    fig.suptitle("Text: which words mattered")
    fig.tight_layout()
    return fig


def _circle(dark=False):
    from ruleofthumb import plot as rot_plot

    rng = np.random.RandomState(0)
    heat = np.zeros((32, 32), np.float32)
    rows, cols = np.ogrid[:32, :32]
    heat[(rows - 16) ** 2 + (cols - 16) ** 2 <= 36] = 1.0
    heat += rng.randn(32, 32).astype(np.float32) * 0.05
    grey = np.zeros((32, 32, 3))
    grey[(rows - 16) ** 2 + (cols - 16) ** 2 <= 36] = 0.9
    fig = rot_plot.saliency(heat, image=grey, size=(3.2, 3.2))
    fig.suptitle("Images: which pixels mattered")
    fig.tight_layout()
    return fig


def _mask(dark=False):
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.6))
    words = [["a", "wonderful", "film"], ["terrible", "pacing"], ["ok"]]
    for ax, title, kind in zip(axes, ["Sentences", "Rectangle + filler", "Mask: True = real"], ["words", "rect", "mask"]):
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.axis("off")
        for r, sent in enumerate(words):
            for c in range(3):
                y = 2 - r
                if kind == "words":
                    if c < len(sent):
                        ax.text(c + 0.5, y + 0.5, sent[c], ha="center", va="center", fontsize=9,
                                bbox={"boxstyle": "round,pad=0.2", "facecolor": "none", "edgecolor": "gray"})
                elif kind == "rect":
                    tok = sent[c] if c < len(sent) else "·"
                    ax.text(c + 0.5, y + 0.5, tok, ha="center", va="center", fontsize=9,
                            bbox={"boxstyle": "round,pad=0.2", "facecolor": "none", "edgecolor": "gray"})
                else:
                    real = c < len(sent)
                    ax.add_patch(plt.Rectangle((c + 0.05, y + 0.05), 0.9, 0.9,
                                               facecolor="#2ca02c" if real else "none", edgecolor="gray"))
                    ax.text(c + 0.5, y + 0.5, "T" if real else "·", ha="center", va="center", fontsize=9)
    fig.suptitle("Padding is explicit: filler means nothing, the mask carries the truth")
    fig.tight_layout()
    return fig


def _reveal(dark=False):
    from ruleofthumb import plot as rot_plot

    good = np.array([0.55, 0.80, 0.95, 1.0, 1.0])
    random = np.array([0.52, 0.55, 0.58, 0.62, 0.66])
    fig = rot_plot.reveal({"RoT order": good, "Random": random}, title="Same answer, fewer inputs")
    return fig


def _limits_order(dark=False):
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.6))
    for ax, sent in zip(axes, ["dog bites man", "man bites dog"]):
        ax.set_title(f'"{sent}"', fontsize=12)
        ax.barh(["dog", "bites", "man"], [0.5, 0.1, -0.4], color=[RED, RED, BLUE])
        ax.set_xlim(-0.6, 0.6)
    fig.suptitle("Word order is invisible: same words, same explanation")
    fig.tight_layout()
    return fig


def _limits_ink(dark=False):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    rows, cols = np.ogrid[:24, :24]
    blob = ((rows - 12) ** 2 + (cols - 12) ** 2 <= 25).astype(float)
    ring = (((rows - 12) ** 2 + (cols - 12) ** 2 <= 64) & ((rows - 12) ** 2 + (cols - 12) ** 2 >= 36)).astype(float)
    ring *= blob.sum() / max(ring.sum(), 1)
    for ax, img, title in zip(axes, [blob, ring], ["tight spot", "wide ring"]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{title}\nsame total ink \u2192 same score", fontsize=10)
        ax.axis("off")
    fig.suptitle("Position is pooled away on raw pixels: use rich features")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _pair("hero", _hero)
    _pair("card-tabular", _card_tabular)
    _pair("card-text", _card_text)
    _pair("card-image", _circle)
    _pair("mask", _mask)
    _pair("reveal", _reveal)
    _pair("limits-order", _limits_order)
    _pair("limits-ink", _limits_ink)
    print("wrote", len(os.listdir(OUT)), "files to", OUT)
