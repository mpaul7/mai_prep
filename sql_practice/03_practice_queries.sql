-- Practice SQL Queries for Data Analyst Preparation
-- These queries cover common data analyst tasks

-- 1. Basic SELECT queries
-- Get all employees
SELECT * FROM employees;

-- Get employees in Sales department
SELECT first_name, last_name, salary 
FROM employees 
WHERE department = 'Sales';

-- 2. Aggregation queries
-- Average salary by department
SELECT department, AVG(salary) as avg_salary
FROM employees
GROUP BY department
ORDER BY avg_salary DESC;

-- Total sales by employee
SELECT e.first_name, e.last_name, SUM(s.sale_amount) as total_sales
FROM employees e
JOIN sales s ON e.employee_id = s.employee_id
GROUP BY e.employee_id, e.first_name, e.last_name
ORDER BY total_sales DESC;

-- 3. Date-based analysis
-- Sales by month
SELECT 
    DATE_FORMAT(sale_date, '%Y-%m') as month,
    COUNT(*) as number_of_sales,
    SUM(sale_amount) as total_revenue
FROM sales
GROUP BY DATE_FORMAT(sale_date, '%Y-%m')
ORDER BY month;

-- 4. JOIN operations
-- Employee details with their sales
SELECT 
    e.first_name,
    e.last_name,
    e.department,
    s.product_name,
    s.sale_amount,
    s.sale_date
FROM employees e
LEFT JOIN sales s ON e.employee_id = s.employee_id
ORDER BY e.last_name, s.sale_date;

-- 5. Subqueries
-- Employees with above-average salary
SELECT first_name, last_name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Top performing employee (highest total sales)
SELECT first_name, last_name
FROM employees
WHERE employee_id = (
    SELECT employee_id
    FROM sales
    GROUP BY employee_id
    ORDER BY SUM(sale_amount) DESC
    LIMIT 1
);

-- 6. Window functions (for advanced practice)
-- Rank employees by salary within department
SELECT 
    first_name,
    last_name,
    department,
    salary,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) as salary_rank
FROM employees;

-- Running total of sales
SELECT 
    sale_date,
    sale_amount,
    SUM(sale_amount) OVER (ORDER BY sale_date) as running_total
FROM sales
ORDER BY sale_date;
