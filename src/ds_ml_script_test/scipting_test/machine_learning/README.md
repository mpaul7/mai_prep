### Machine Learning — Linear Regression (Scripting Test)

This module starts ML scripting with linear regression: training, evaluation, and simple cross-validation.

#### Files
- `linear_regression_example.py`: synthetic data, train/test split, metrics, CV R^2
- `hackerrank_questions.md`: 3 prompts (metrics, CV, holdout prediction)
- `solutions.py`: CLI for Q1–Q3

#### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

#### Run example
```bash
python src/ds_ml_script_test/scipting_test/machine_learning/linear_regression_example.py
```

#### Run solutions
```bash
# Q1: train/test metrics
python src/ds_ml_script_test/scipting_test/machine_learning/solutions.py q1 --train train.csv --output metrics.csv

# Q2: 5-fold CV R^2
python src/ds_ml_script_test/scipting_test/machine_learning/solutions.py q2 --train train.csv --output cv.csv

# Q3: predict on holdout
python src/ds_ml_script_test/scipting_test/machine_learning/solutions.py q3 --train train.csv --holdout holdout.csv --output preds.csv
```

#### Notes
- Ensure the training CSV has `y` column as target and the rest are features.
- Features should be numeric; handle preprocessing upstream if needed.




