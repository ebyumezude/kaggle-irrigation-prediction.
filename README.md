# Kaggle Irrigation Need Prediction

**Author:** Ebelechukwu Umezude ([ebyumezude](https://github.com/ebyumezude))

A machine learning project for the AIS COMPETITION 2026 Kaggle exercise.
It predicts whether a field has **Low**, **Medium** or **High** irrigation need.
The model is a scikit-learn histogram gradient boosting classifier.

## Files

- `irrigation_model.ipynb`: the complete workflow with recorded run outputs.
- `submission.csv`: 270,000 test predictions, with columns `id,Irrigation_Need`.
- `requirements.txt`: Python dependencies.

## Dataset

The original training data has 630,000 rows and 21 columns. The test data has
270,000 rows and 20 columns. `Irrigation_Need` is the target. `id` is excluded
from the model, leaving 19 predictors: eight categorical and eleven numeric.
Training labels are Low: 369,917, Medium: 239,074, and High: 21,009.

Get `train.csv` and `test.csv` from the course's Kaggle competition using an
account with access. They are not redistributed in this repository. Place both
files next to the notebook. The submission includes predictions, not test labels.

## Method

1. Load the data and inspect its dimensions and target counts.
2. Exclude `id` and separate the target.
3. Make a stratified 80/20 training/validation split with random_state=42.
   The split contains 504,000 training and 126,000 validation examples.
4. Use OrdinalEncoder for categorical inputs. Tell the classifier which encoded
   features are categorical; numeric features pass through unchanged. The
   encoder is fitted within the training pipeline, not on validation data.
5. Train HistGradientBoostingClassifier with balanced sample weights, helping
   the rare High class contribute to learning. Main settings: learning_rate=0.10,
   max_iter=200, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True.
6. Evaluate against a majority-class baseline using balanced accuracy,
   ordinary accuracy, a classification report and a confusion matrix.
7. Fit a fresh copy of the same pipeline on the full training input (with
   internal early stopping still enabled) and generate the submission.

Balanced accuracy averages recall across the three classes, giving each class
equal importance despite the class imbalance.

## Recorded validation results

| Metric | Result |
|---|---:|
| Majority-class baseline balanced accuracy | 0.3333 |
| Model balanced accuracy | 0.9728 |
| Model ordinary accuracy | 0.9842 |
| Low recall | 0.9949 |
| Medium recall | 0.9703 |
| High recall | 0.9534 |

These are results from the notebook's validation split, **not a confirmed Kaggle
leaderboard score**. There is one validation split, so the score does not prove
performance under a different data distribution.

## Run locally

The original notebook records Python 3.11.15. Original dependency versions were
not recorded; requirements.txt lists dependencies rather than an exact lockfile.
Results can vary across library versions and environments.

In a terminal in this repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open irrigation_model.ipynb, select the environment's Python kernel and run the
cells in order. Training may take several minutes. The last cell writes
`submission.csv` in the working directory, replacing an existing file of that
name. The saved notebook outputs can be viewed without rerunning training.

## Submission validation

The supplied CSV was checked for the expected headers, 270,000 rows, unique IDs
630000 through 899999 in order, and allowed nonempty labels. Prediction counts:
Low 159,734; Medium 100,677; High 9,589. These format checks do not establish
prediction accuracy on the hidden test labels.

## Assistance

The workflow and repository documentation were prepared with ChatGPT assistance.
