# Core ideas

Six terms, each in one paragraph. Everything else in these docs builds
on them.

**Black box.** Anything that maps inputs to answers: `y = model(X)`.
A sklearn classifier, an LLM sentiment head, an image network. RoT never
opens it — it only watches what it answers.

**Stand-in (surrogate).** A deliberately simple model fitted to copy the
black box's answers on your data. You create one per modality with
`fit_tabular`, `fit_text`, or `fit_image` (or `fit`, which picks from
the input shape). The returned `Explainer` wraps it.

**Importance.** One signed number per input piece (a column, a word, a
pixel). Positive means evidence *toward* the predicted answer, negative
means evidence *against* — the red/blue convention in every plot. For
two-answer tasks you get the numbers for the second answer directly;
for more answers you get one set per answer.

**Order.** `get_order` sorts each row's pieces most-important-first by
absolute importance. It answers "what should I look at first?"

**Reveal curve.** `ordered_predict` uncovers pieces in that order and
re-scores the answer at each step; `score_ordering` summarises how
faithful the ranking is (usually accuracy per step). A good explanation
keeps the right answer after revealing very little.

**Mask.** Batches are rectangles, but sentences and pictures vary in
length and size — so short rows get meaningless filler plus a boolean
mask saying what is real (`True` = real word/pixel). The filler value
never matters; the mask carries the truth. Padded slots always score
exactly zero and rank last (reported as `-1` in orders).

```{image} _static/figures/mask.light.png
:class: only-light
:alt: Three sentences padded into a rectangle with a green mask marking real tokens
```

```{image} _static/figures/mask.dark.png
:class: only-dark
:alt: Three sentences padded into a rectangle with a green mask marking real tokens
```

**Agreement.** `exp.train_agreement_` is the stand-in's accuracy against
the black box's answers on the training inputs. It is optimistic
(in-sample) — a tripwire, not a verdict. Near 1: proceed. Below ~0.75
the fit warns you, and the numbers may be gibberish. [Limits](capacity.md)
explains the common causes.
