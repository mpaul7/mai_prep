### Reusable Functions for Data Processing

Demonstrates writing small, testable functions you can reuse in data tasks.

#### Files
- `functions_example.py`: examples (coalesce, normalize, zscore, aggregate)
- `hackerrank_questions.md`: 3 function-focused prompts
- `solutions.py`: CLI solutions for Q1–Q3

#### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

#### Run example
```bash
python src/ds_ml_script_test/python_skills/functions/functions_example.py
```

#### Run solutions
```bash
# Q1 coalesce_str
python src/ds_ml_script_test/python_skills/functions/solutions.py q1 --input values.txt --fallback unknown --output out.txt

# Q2 normalize_text
python src/ds_ml_script_test/python_skills/functions/solutions.py q2 --input text.txt --output normalized.txt

# Q3 aggregate_by
python src/ds_ml_script_test/python_skills/functions/solutions.py q3 --input data.csv --output agg.csv
```

