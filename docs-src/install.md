# Install

Requires Python ≥ 3.9.

**Stable (from PyPI)** — what most people want:

```bash
pip install ruleofthumb-rot
```

Add only the extras you need:

| Extra | You want to | Pulls in |
|---|---|---|
| `ruleofthumb-rot[text]` | explain sentences from raw strings | `transformers` |
| `ruleofthumb-rot[image]` | explain image files via a backbone | `torchvision`, `Pillow`, `captum` |
| `ruleofthumb-rot[plot]` | draw explanations | `matplotlib`, `wordcloud`, `shap` |

```bash
pip install "ruleofthumb-rot[text,image,plot]"
```

**Latest (unreleased, from source):**

```bash
pip install git+https://github.com/KaiRawal/Rule-of-Thumb.git
```

Check it worked:

```python
import ruleofthumb as rot
print(rot.__version__)
```

:::{note}
**Docs versions:** `stable` at `/en/stable/` matches the latest PyPI
release; `latest` at `/en/latest/` tracks `main` and may describe
unreleased APIs. Use the version flyout (lower-right) to switch.
:::

Next: the 5-minute [Quickstart](quickstart.md).
