### Analysis with pandas (aggregating, filtering, summarizing)

This module focuses on analysis operations you’ll encounter in tests: filtering, aggregations, rolling windows, crosstabs, and ranking.

#### Files
- `analysis_example.py`: end-to-end analysis demo
- `hackerrank_questions.md`: analysis-style prompts
- `solutions.py`: CLI solutions to Q1–Q3

#### Setup
From project root:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

#### Run example
```bash
python src/ds_ml_script_test/python_skills/analysis/analysis_example.py
```

#### Run solutions
```bash
# Q1 Category Contribution by Region
python src/ds_ml_script_test/python_skills/analysis/solutions.py q1 --input input.csv --output contribution.csv

# Q2 Rolling Weekly Revenue
python src/ds_ml_script_test/python_skills/analysis/solutions.py q2 --input weekly.csv --output rolling.csv

# Q3 Top-N Categories by Region
python src/ds_ml_script_test/python_skills/analysis/solutions.py q3 --input data.csv --top_n 2 --output topn.csv
```

#### Notes
- Ensure inputs include required columns as described in `hackerrank_questions.md`.
- Sort before rolling operations to guarantee correct window semantics.

