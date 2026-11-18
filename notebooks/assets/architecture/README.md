# Figure inventory

Exactly 12 distinct instructional figures are provided in two generated editions. `notebooks/pluto_e2e_walkthrough.ipynb` uses relative PNG links for GitHub and local repository viewing. `notebooks/pluto_e2e_walkthrough_offline.ipynb` embeds self-contained SVG attachments for use when copied alone. Their code cells and lesson content are identical. The opening figure includes generated conceptual artwork; all computational layers, geometry, labels and dependencies remain editable in `scripts/notebook_figures.py`. The generated artwork and its exact prompt are retained in `../illustrations/`.

| Figure | Cell anchor | Learning question | Source |
| --- | --- | --- | --- |
| overview | `intro` | How does a scene become a selected plan? | `pluto/model.py` |
| data-flow | `data` | Which adjacent history pairs remain valid? | `pluto/data.py` |
| history-pyramid | `history` | How do short and long temporal contexts combine? | `pluto/model.py` |
| scene-tokens | `scene` | How do different object types share a feature width? | `pluto/model.py` |
| query-grid | `arch` | Which queries interact in each attention stage? | `pluto/model.py` |
| shapes | `assembly` | Where do the batch, route, mode and time axes go? | `pluto/model.py` |
| loss | `objective` | How does distance to unsafe space penalize a plan? | `pluto/losses.py` |
| optimizer | `optimization` | What happens between a loss and a parameter update? | `pluto/losses.py` |
| contrastive | `cil` | Which changed views can still use their old targets? | `pluto/objective.py` |
| inference | `evaluation` | How are selection and geometric error different? | `pluto/planner.py` |
| bicycle | `simulation` | How does a controller track a reference plan? | `pluto/simulation.py` |
| checkpoint | `checkpoint` | What must match before training can resume? | `pluto/train.py` |

Each figure follows its explanation and precedes the relevant probe or definition. The full model is defined locally in `model-definitions`; it follows the three encoder/decoder mechanism illustrations. Repeated attention blocks share one worked figure. The package's optional external rule evaluator appears beside learned selection; it is not implemented in the notebook.

Regenerate with `python scripts/make_notebook.py` and `python scripts/render_figures.py`. Run `python scripts/check_notebook.py` and `python scripts/run_notebook.py`. Analytical arithmetic and model output/loss/gradient parity are checked in the tests. See `inspection.md` for visual inspection evidence and limits. Code outputs are deliberately empty in version control.
