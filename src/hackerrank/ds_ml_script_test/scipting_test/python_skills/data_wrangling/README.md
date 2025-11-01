### Data Wrangling with pandas

This example demonstrates an end-to-end data cleaning and transformation workflow using pandas, similar to tasks you might see in HackerRank Data Analyst or ML scripting tests.

#### Files
- `data_wrangling_example.py`: main example script
- `hackerrank_questions.md`: prompts styled like HackerRank
- `solutions.py`: solutions with CLI subcommands (`q1`, `q2`, `q3`)

#### What it covers
- **Synthetic messy data**: create `orders` and `customers` with issues
- **Missing values**: imputation for quantity, safe parsing for price/date
- **Duplicates**: remove duplicated rows
- **Type conversions**: currency strings to float; text normalization
- **Text cleaning**: trim, collapse spaces, lowercase, recode categories
- **Date handling**: parse with coercion, extract year/month/day/DoW
- **Feature engineering**: revenue, high-value flags, phone normalization
- **Aggregations**: customer revenue and order counts
- **Window functions**: running revenue per customer
- **Joins**: enrich orders with customer attributes
- **Reshaping**: pivot and melt monthly category revenue
- **Outliers**: cap revenue via IQR method
- **Export**: write CSV (always) and Parquet (if `pyarrow` available)

#### Setup
From the project root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pyarrow
```

#### Run

```bash
python src/ds_ml_script_test/python_skills/data_wrangling/data_wrangling_example.py
```

Outputs are written to the `results/` directory:
- `fact_orders.csv` and `.parquet`
- `customer_revenue.csv`
- `monthly_category_revenue.csv`

#### Tips for tests
- Read the prompt carefully; confirm required columns and output ordering.
- Use `errors="coerce"` when parsing unknown formats, then handle `NaN/NaT`.
- Prefer vectorized operations over row-wise `apply` for performance.
- Validate merges with `validate="many_to_one"` or similar.

#### Hackerrank-style practice
- See `hackerrank_questions.md` for problem statements.
- Run solutions:

```bash
# Q1 Clean Customer Directory
python src/ds_ml_script_test/python_skills/data_wrangling/solutions.py q1 --input customers.csv --output cleaned_customers.csv

# Q2 Monthly Revenue from Orders
python src/ds_ml_script_test/python_skills/data_wrangling/solutions.py q2 --input orders.csv --output monthly_revenue.csv

# Q3 Top-N Customers by Revenue
python src/ds_ml_script_test/python_skills/data_wrangling/solutions.py q3 --orders orders.csv --customers customers.csv --top_n 5 --output top_customers.csv
```

