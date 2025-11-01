### Basic Statistics (correlation, probability)

This module covers essential stats tasks you’ll face in scripting tests.

#### Files
- `statistics_example.py`: demo of correlation, covariance, z-scores, binomial, conditional probability
- `hackerrank_questions.md`: 3 practice prompts
- `solutions.py`: CLI solutions for Q1–Q3

#### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

#### Run example
```bash
python src/ds_ml_script_test/python_skills/statistics/statistics_example.py
```

#### Run solutions
```bash
# Q1 Pearson Correlation
python src/ds_ml_script_test/python_skills/statistics/solutions.py q1 --input xy.csv

# Q2 Binomial Tail Probability
python src/ds_ml_script_test/python_skills/statistics/solutions.py q2 --n 10 --k 4 --p 0.3

# Q3 Conditional Probability from Contingency
python src/ds_ml_script_test/python_skills/statistics/solutions.py q3 --input data.csv --target A
```

