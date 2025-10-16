### Python Data Structures

Fundamentals used in scripting tests: lists, dicts, sets, tuples, stacks/queues, heaps.

#### Files
- `data_structures_example.py`: core operations and small demos
- `hackerrank_questions.md`: 3 practice prompts
- `solutions.py`: CLI solutions for Q1–Q3

#### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

#### Run example
```bash
python src/ds_ml_script_test/python_skills/data_structures/data_structures_example.py
```

#### Run solutions
```bash
# Q1 Anagram Groups
python src/ds_ml_script_test/python_skills/data_structures/solutions.py q1 --input words.txt --output groups.txt

# Q2 Merge Dictionaries with Sum
python src/ds_ml_script_test/python_skills/data_structures/solutions.py q2 --inputs d1.json d2.json d3.json --output merged.json

# Q3 K Most Frequent Items
python src/ds_ml_script_test/python_skills/data_structures/solutions.py q3 --input items.txt --k 3 --output topk.csv
```

