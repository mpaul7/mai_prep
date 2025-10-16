### Hackerrank-style Data Wrangling Questions (pandas)

These prompts are representative of data wrangling scripting questions you might encounter. Inputs are CSV data unless otherwise specified.

#### Q1. Clean Customer Directory
You are given a CSV of customer data with the following columns:
- `name` (string, may include extra spaces and inconsistent case)
- `city` (string, may include extra spaces and inconsistent case, can be missing)
- `state` (string, may include extra spaces and inconsistent case, can be missing)
- `phone` (string, may include symbols and spaces, can be missing)

Task:
- Standardize text fields by trimming, collapsing internal whitespace to one space, and lowercasing.
- Normalize `phone` to digits only (remove all non-digit chars). If empty after cleaning, leave it blank.

Output CSV columns (in order):
- `name_clean`, `city_clean`, `state_clean`, `phone_digits`

Constraints:
- Missing `city`/`state` become empty strings in the output.

Example input:
```
name,city,state,phone
  Alice  Smith , New   York , NY , (555) 111-2222
Bob Jones, san  francisco , ca, +1 555 333 4444
Carol, , ,  
```

Expected output:
```
name_clean,city_clean,state_clean,phone_digits
alice smith,new york,ny,5551112222
bob jones,san francisco,ca,15553334444
carol,,, 
```

---

#### Q2. Monthly Revenue from Orders
You are given a CSV of orders with the following columns:
- `order_id` (int)
- `order_date` (string in YYYY-MM-DD or malformed)
- `price` (string currency like "$ 12.34", may be "N/A")
- `quantity` (int, can be missing)

Task:
- Parse `price` into a numeric value. Invalid or missing becomes `NaN`.
- Parse `order_date` into a date, coercing invalid to `NaT`. Drop rows where date or price is missing/invalid.
- For `quantity`, impute missing with the median quantity of valid rows.
- Compute `revenue = price * quantity`.
- Aggregate total monthly revenue by year and month.

Output CSV columns (in order):
- `year`, `month`, `monthly_revenue`

---

#### Q3. Top-N Customers by Revenue
Given two CSVs:
1) Orders with columns: `order_id`, `customer_id`, `order_date`, `price`, `quantity`
2) Customers with columns: `customer_id`, `name`

Task:
- Parse `price` and `order_date` as in Q2. Drop rows with invalid date/price.
- Fill missing `quantity` with the median.
- Compute revenue and total per customer.
- Return the top `N` customers by total revenue (descending). If two customers tie, order by `customer_id` ascending.

Output CSV columns (in order):
- `customer_id`, `name`, `total_revenue`, `order_count`

Parameters:
- `N` will be provided as a command-line argument.



