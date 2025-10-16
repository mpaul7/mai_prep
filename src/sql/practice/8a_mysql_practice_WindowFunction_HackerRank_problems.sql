
drop table if exists sales;
select * from sales;
CREATE TABLE sales (
    sale_id INT,
    emp_name VARCHAR(50),
    department VARCHAR(50),
    sale_amount DECIMAL(10,2),
    sale_date DATE
);

INSERT INTO sales VALUES
(1, 'Alice', 'Electronics', 5000, '2024-01-01'),
(2, 'Bob', 'Electronics', 3000, '2024-01-02'),
(3, 'Charlie', 'Electronics', 7000, '2024-01-03'),
(4, 'David', 'Clothing', 4000, '2024-01-02'),
(5, 'Eve', 'Clothing', 6000, '2024-01-04'),
(6, 'Frank', 'Clothing', 6000, '2024-01-05'),
(7, 'Grace', 'Home', 2000, '2024-01-01'),
(8, 'Hank', 'Home', 3500, '2024-01-02'),
(9, 'Ivy', 'Home', 4500, '2024-01-03'),
(10, 'Jack', 'Home', 5000, '2024-01-04');

-- ==========================================================
-- Q1. Rank employees by sales amount within each department
-- Assign ranks to employees in each department based on their sale amount (highest first).
-- If two employees have the same amount, they share the same rank, and the next rank is skipped.
-- ===============================================================================================
SELECT
    emp_name,
    department,
    sale_amount,
    RANK() OVER (
        PARTITION BY department
        ORDER BY sale_amount DESC
    ) AS dept_rank
FROM sales;


-- ==================================================================
-- Q2. Find top 2 employees by sales amount per department
-- Return only the top 2 performing employees from each department.
-- ==================================================================
WITH ranked_sales AS (
    SELECT
        emp_name,
        department,
        sale_amount,
        ROW_NUMBER() OVER (
            PARTITION BY department
            ORDER BY sale_amount DESC
        ) AS rn
    FROM sales
)
SELECT emp_name, department, sale_amount
FROM ranked_sales
WHERE rn <= 2;

-- =================================================
-- Q3. Compare each employee’s sale with the previous employee in their department
-- Show the difference between each employee’s sale and the previous employee’s sale in the same department.
-- ================================================================================
SELECT
    emp_name,
    department,
    sale_amount,
    LAG(sale_amount) OVER (
        PARTITION BY department
        ORDER BY sale_amount DESC
    ) AS prev_sale,
    sale_amount - LAG(sale_amount) OVER (
        PARTITION BY department
        ORDER BY sale_amount DESC
    ) AS diff_from_prev
FROM sales;

-- ========================================================================================
-- Q4. Compute running total of sales by department
-- Show the cumulative (running) sales total for each department, ordered by sale amount.
-- =======================================================================================
SELECT
    emp_name,
    department,
    sale_amount,
    SUM(sale_amount) OVER (
        PARTITION BY department
        ORDER BY sale_amount
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM sales;

-- ==========================================================
-- Q5. Show each employee’s next sale (by date) in their department
-- Use LEAD() to get the next sale amount after the current sale, within the same department.
-- =================================================================
SELECT
    emp_name,
    department,
    sale_date,
    sale_amount,
    LEAD(sale_amount) OVER (
        PARTITION BY department
        ORDER BY sale_date
    ) AS next_sale
FROM sales;
