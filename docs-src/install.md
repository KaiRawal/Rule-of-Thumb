# Install

You will need Python 3.9 or newer — everything else arrives with the
package you pick below.

**Stable (from PyPI)** — the right choice for almost everyone:

```bash
pip install ruleofthumb-rot
```

The base install stays deliberately small (`numpy` plus `torch`), and
each kind of data unlocks with its own extra, so you only download
what you will actually use:

| Extra | Choose it when you want to | It brings |
|---|---|---|
| `ruleofthumb-rot[text]` | explain sentences from raw strings | `transformers` |
| `ruleofthumb-rot[image]` | explain image files via a backbone | `torchvision`, `Pillow`, `captum` |
| `ruleofthumb-rot[plot]` | draw your explanations | `matplotlib`, `wordcloud`, `shap` |

```bash
pip install "ruleofthumb-rot[text,image,plot]"
```

**Latest (unreleased, from source):** only if you are following
development closely and are comfortable with moving targets:

```bash
pip install git+https://github.com/KaiRawal/Rule-of-Thumb.git
```

A quick check that everything arrived safely:

```python
import ruleofthumb as rot
print(rot.__version__)
```

:::{note}
**Docs versions:** `stable` at `/en/stable/` matches the latest PyPI
release; `latest` at `/en/latest/` tracks `main` and may describe
features that are not released yet. Use the version flyout
(lower-right) to switch between them.
:::

With that in place, the 5-minute [Quickstart](quickstart.md) is the
natural next stop.
