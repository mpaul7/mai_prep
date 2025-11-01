-- A window function performs a calculation across a set of table rows that are related to the current row. 
-- They don’t collapse rows like GROUP BY — each row remains in the output.

-- setup table 
select * from sales;
CREATE TABLE sales (
    id INT,
    emp_name VARCHAR(50),
    department VARCHAR(50),
    sale_amount DECIMAL(10,2)
);

INSERT INTO sales VALUES
(1, 'Alice', 'Electronics', 5000),
(2, 'Bob', 'Electronics', 3000),
(3, 'Charlie', 'Electronics', 7000),
(4, 'David', 'Clothing', 4000),
(5, 'Eve', 'Clothing', 6000),
(6, 'Frank', 'Clothing', 6000);

-- ======================
-- ROW_NUMBER()
-- ======================
-- Assigns a unique sequential number to each row within a partition.


SELECT
    emp_name,
    department,
    sale_amount
FROM sales;

SELECT
    emp_name,
    department,
    sale_amount,
    ROW_NUMBER() OVER (
        PARTITION BY department
        ORDER BY sale_amount DESC
    ) AS row_num
FROM sales;

-- ========================
-- RANK()
-- =======================

-- Assigns a rank starting from 1 within each partition.
-- Ties share the same rank, but the next rank skips numbers.

SELECT
    emp_name,
    department,
    sale_amount,
    RANK() OVER (
        PARTITION BY department
        ORDER BY sale_amount DESC
    ) AS rnk
FROM sales;

-- ===========================
-- DENSE_RANK()
-- ===========================
SELECT
    emp_name,
    department,
    sale_amount,
    DENSE_RANK() OVER (
        PARTITION BY department
        ORDER BY sale_amount DESC
    ) AS dense_rnk
FROM sales;

-- =========================
-- LAG()
-- Accesses previous row’s value (within partition).
-- =========================================

SELECT
    emp_name,
    department,
    sale_amount,
    LAG(sale_amount, 1) OVER (
        PARTITION BY department
        ORDER BY sale_amount DESC
    ) AS prev_sale
FROM sales;

-- ==============================================
-- LEAD()
-- Accesses next row’s value (within partition).
-- ==============================================

SELECT
    emp_name,
    department,
    sale_amount,
    LEAD(sale_amount, 1) OVER (
        PARTITION BY department
        ORDER BY sale_amount DESC
    ) AS next_sale
FROM sales;


-- Top N per group
SELECT * FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY department ORDER BY sale_amount DESC) AS rn
    FROM sales
) ranked
WHERE rn <= 2;

-- Running Total
SELECT
    emp_name,
    department,
    sale_amount,
    SUM(sale_amount) OVER (
        PARTITION BY department
        ORDER BY sale_amount
    ) AS running_total
FROM sales;

-- Compare employee’s sale to department average
SELECT
    emp_name,
    department,
    sale_amount,
    AVG(sale_amount) OVER (PARTITION BY department) AS dept_avg
FROM sales;

-- Rank employees across all departments
SELECT
    emp_name,
    department,
    sale_amount,
    RANK() OVER (ORDER BY sale_amount DESC) AS company_rank
FROM sales;
