# DSC PJATK Catalogue Quality Challenge: Team 2

**Team:** Team 2. **Members:** Piotr Egeman, Maciej Gruza.

We rank marketplace listings by the probability that they contain a controlled data error (wrong category,
package price, shipping weight or main-item quantity). The model checks each listing against its own text and
against what is normal for the same product.

| Submission | Notebook | Pooled OOF ROC-AUC | OOF AP | Public LB ROC-AUC |
|---|---|---|---|---|
| **Final model** (`ref`) | `02_model.ipynb` | **0.9688** | 0.926 | **0.97256** |
| Earlier submission (first version, before the leakage fixes) | `02b_model_first_version.ipynb` | 0.9683 | 0.928 | 0.96594 |
| Organisers' baseline | (starter notebook) | 0.7082 | 0.571 | |

OOF scores use the organisers' five seller-disjoint folds. The full write-up is in the solution report (`2_report.pdf`).

## Repository contents

| File | Content |
|---|---|
| `01_eda.ipynb` | Exploratory analysis: class balance, quantity mismatch, price/weight tails relative to the product, missingness, sellers, rule precision and recall |
| `02_model.ipynb` | **Final model.** Full pipeline (parsing, features, seller-disjoint CV, LightGBM, inference); writes `submission.csv`, OOF predictions and a CV report |
| `02b_model_first_version.ipynb` | Our earlier submission, original code unchanged. Kept so that both selected submissions can be reproduced; its known validation issues are listed at the top of the notebook |
| `03_experiments.ipynb` | Numbers behind Sections 3 and 5 of the report: EDA tables, the two controlled feature comparisons and the five-error OOF analysis |
| `catalogue_pipeline.py` | The pipeline as a Python module. `02_model.ipynb` and `03_experiments.ipynb` contain the same code inline, so they run on Kaggle without this file |
| `results/` | OOF predictions with fold assignments (`oof_*.csv`), CV reports (`cv_*.json`), comparison tables and figure |

The competition data is not included, because redistribution is not allowed.

## How to reproduce

**On Kaggle (the reference environment):**

1. Open the competition and create a new notebook (Code, then New Notebook), so that the competition data is attached.
2. File, then Import Notebook, and select `02_model.ipynb`.
3. Session options: Accelerator None, Internet Off.
4. Run All. The run takes about 30 seconds and writes `/kaggle/working/submission.csv`, `cv_ref.json` and `oof_ref.csv`.

The same steps reproduce the earlier submission (`02b_model_first_version.ipynb`, about 40 seconds) and the
experiments (`03_experiments.ipynb`, about 3 minutes). `01_eda.ipynb` reads the data from the Kaggle input path.

**Locally:**

```
pip install -r requirements.txt
python catalogue_pipeline.py --data-dir path/to/competition/data --out results            # final model
python catalogue_pipeline.py --data-dir path/to/competition/data --out results --config c1_first_pcs
python catalogue_pipeline.py --data-dir path/to/competition/data --out results --config c2_no_anomaly
```

Each run asserts that the fold assignment matches the organisers' baseline (fold hash `dda65b9d…`) and records
the library versions and the SHA-256 of the data files.

## Validation and reproducibility

- Folds: the organisers' seller-disjoint assignment, SHA-256(`42:<seller_id>`), first 16 hex digits, modulo 5.
- Every learned statistic (product-phrase purity and dominant category, modal pack size, price and weight medians
  and MADs, categorical levels) is fitted on the training folds only.
- The number of trees is fixed at 400. There is no early stopping on the held-out fold.
- LightGBM runs with `deterministic=True`, 4 threads and a fixed seed per fold. Two runs give identical predictions,
  and a local run with the versions below reproduces the Kaggle OOF scores to six decimals.

## Requirements

Python 3.13, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, lightgbm 4.6.0 (the Kaggle environment versions, see
`requirements.txt`). The notebooks also use matplotlib.

## AI assistance

We used Claude (Anthropic), Gemini (Google) and DeepSeek to write, explain, review and debug code and to draft the
report. No AI tool supplied labels, predictions or per-listing features. Details are in the report.

## License

MIT
