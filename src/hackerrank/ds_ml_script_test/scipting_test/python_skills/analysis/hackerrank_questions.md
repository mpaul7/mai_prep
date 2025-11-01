### Hackerrank-style Analysis Questions (pandas)

#### Q1. Category Contribution by Region
Input CSV has columns: `region`, `category`, `revenue`.

Task:
- Aggregate total revenue by `region` and `category`.
- Compute each category's share of regional revenue as a percentage with 2 decimals.
- Output rows sorted by `region` ascending, then `share_pct` descending.

Output columns: `region`, `category`, `revenue_sum`, `share_pct`

---

#### Q2. Rolling Weekly Revenue
Input CSV has columns: `region`, `week` (1-52), `revenue` (weekly total, may contain duplicates for same region-week from different sources).

Task:
- Sum `revenue` within each `(region, week)`.
- Compute `rolling_4w` as the sum of the last 4 weeks' revenue by `region` (include current week, min periods = 1).
- Output sorted by `region`, then `week`.

Output columns: `region`, `week`, `weekly_revenue`, `rolling_4w`

---

#### Q3. Top-N Categories by Region
Input CSV has columns: `region`, `category`, `revenue`.

Task:
- Compute total `revenue` per `(region, category)`.
- Rank categories within each region by revenue (descending) using first-occurrence tie-break.
- Return top `N` per region. `N` is provided as a CLI argument.

Output columns: `region`, `category`, `revenue_sum`, `rank`


