# Migration notes

Breaking changes across the 0.x line. The package is pre-1.0: breaking
API changes are acceptable, each one recorded here.

## Migrating from v0.1 sentinel padding

v0.2 removed the implicit `-1` sentinel: **no fill value has special
meaning any more**. Padding is now always explicit via a validity mask
(`True` = real token/pixel). Code changes required:

```python
# v0.1 (implicit): pad with -1, the model inferred padding from the data
x[:, n_tokens:] = -1.0
rot = TextRuleOfThumb(y, x)
exp = rot.get_explanation(x)

# v0.2 (explicit): keep any pad value you like, but pass the mask yourself
x[:, n_tokens:] = 0.0                                # any value works now
lengths = torch.tensor([n_tokens] * len(x))          # or a (N, T) boolean mask
rot = TextRuleOfThumb(y, x, lengths=lengths)         # or attention_mask=...
exp = rot.get_explanation(x, lengths=lengths)

# Migrating an existing -1-padded array? Rebuild its mask in one line:
from ruleofthumb.text import sentinel_mask
mask = sentinel_mask(x_old)                          # True where tokens are real
```

Without a mask every position is treated as real data — padded positions
are no longer masked implicitly.

## Migrating from the v0.2.x wrapper classes

v0.2.10 replaced the `RuleOfThumb` / `TextRuleOfThumb` wrapper classes
with one facade: the `Explainer` class plus `fit` / `fit_tabular` /
`fit_text` / `fit_image` factories. Training arguments are unchanged;
construction moves from constructors to factories:

```python
# v0.2.x
from ruleofthumb import RuleOfThumb, TextRuleOfThumb
rot = RuleOfThumb(y_outputs=y, x_inputs=X)
rot = TextRuleOfThumb(y_outputs=y, x_inputs=x, lengths=lengths)

# v0.2.10+
import ruleofthumb
exp = ruleofthumb.fit(y_outputs=y, x_inputs=X)                    # modality auto-detected
exp = ruleofthumb.fit_text(y_outputs=y, x_inputs=x, lengths=lengths)
```

The fitted explainer exposes the same `get_explanation` semantics, plus
delegating `get_order` / `ordered_predict` / `score_ordering` / `score` /
`predict` methods (previously reached via the private `_explainer_model`
attribute). The raw models (`RoT`, `RoTText`, `RoTImage`) are unchanged.

## v0.0.1 version reset

The first public release reset the version from the internal `0.2.19` to
`0.0.1` to mark pre-alpha status and reserve the PyPI name. No behaviour
changed; the `0.2.x` history is preserved in `ToDo.md`'s changelog.
