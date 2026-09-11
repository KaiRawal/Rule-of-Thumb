# Migration notes

Breaking changes across the 0.x line. The package is pre-1.0: breaking
API changes are acceptable, each one recorded here.

## Migrating to v0.0.2

v0.0.2 removes the deprecated padding spellings: `lengths=` and
`attention_mask=` kwargs are gone from every entry point — pass a single
boolean validity mask as `mask=` (build it from lengths with
`lengths_to_mask`), and the `sentinel_mask` helper is deleted. Save files
now load only under the exact package version that wrote them — refit and
re-save after upgrading.

## Migrating from v0.1 sentinel padding

v0.2 removed the implicit `-1` sentinel: **no fill value has special
meaning any more**. Padding is now always explicit via a validity mask
(`True` = real token/pixel). Code changes required:

```python
# v0.1 (implicit): pad with -1, the model inferred padding from the data
x[:, n_tokens:] = -1.0
exp = rot.fit_text(y_outputs=y, x_inputs=x)

# v0.2 (explicit): keep any pad value you like, but pass the mask yourself
x[:, n_tokens:] = 0.0                                # any value works now
mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool)
mask[:, :n_tokens] = True                            # or lengths_to_mask(lengths, T)
exp = rot.fit_text(y_outputs=y, x_inputs=x, mask=mask)
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
from ruleofthumb.text import lengths_to_mask
exp = ruleofthumb.fit(y_outputs=y, x_inputs=X)                    # modality auto-detected
exp = ruleofthumb.fit_text(y_outputs=y, x_inputs=x, mask=lengths_to_mask(lengths, x.shape[1]))
```

The fitted explainer exposes the same `get_explanation` semantics, plus
delegating `get_order` / `ordered_predict` / `score_ordering` / `score` /
`predict` methods (previously reached via the private `_explainer_model`
attribute). The raw models (`RoT`, `RoTText`, `RoTImage`) are unchanged.

## v0.0.1 version reset

The first public release reset the version from the internal `0.2.19` to
`0.0.1` to mark pre-alpha status and reserve the PyPI name. No behaviour
changed; the `0.2.x` history is preserved in `ToDo.md`'s changelog.
