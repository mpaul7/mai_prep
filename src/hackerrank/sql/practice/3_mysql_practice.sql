
show tables;
drop table if exists employees;
drop table if exists departments;


-- ========================
-- Create table 
-- ========================
CREATE TABLE employees (
    employee_id INT AUTO_INCREMENT PRIMARY KEY,   -- Integer + primary key
    first_name VARCHAR(50) NOT NULL,             -- String, not null
    last_name VARCHAR(50) NOT NULL,
    salary DECIMAL(10,2),                        -- Decimal for salary
    dept_name VARCHAR(50),
    dept_id int                                     -- String for department
);

INSERT INTO employees 
(first_name, last_name, salary, dept_name, dept_id) 
VALUES 
('Alice', 'Johnson', 60000, 'HR', 1),
('Alex', 'Chan', 62000, 'HR', 1),
('Alan', 'Chad', 66000, 'HR', 1);

INSERT INTO employees 
(first_name, last_name, salary, dept_name, dept_id) 
VALUES 
('Ban', 'Brown', 75000, 'Engineering', 2),
('Bliss', 'Tan', 78000, 'Engineering', 2);


INSERT INTO employees 
(first_name, last_name, salary, dept_name, dept_id)
VALUES 
('Bob', 'Smith', 75000, 'Engineering', 2),
('Bib', 'Smyth', 78000, 'Engineering', 2),
('Charlie', 'Lee', 50000, 'Marketing', 3),
('Chad', 'Lee', 55000, 'Marketing', 3),
('Diana', 'Brown', 70000, 'Sales', 4),
('Dan', 'Brew', 75000, 'Sales', 4);

-- =====================================================
-- Subquery in WHERE Clause
-- =====================================================
SELECT first_name, last_name, salary
FROM employees
WHERE salary > 0;

SELECT first_name, last_name, salary
FROM employees
WHERE salary = (SELECT MAX(salary) FROM employees);

-- =========================
-- Subquery in SELECT Clause
-- ==========================
SELECT first_name, last_name, salary, (SELECT AVG(salary) FROM employees) AS avg_salary_all
FROM employees;

-- =================
-- Subquery in FROM
-- =================
select * from employees;

SELECT dept_id, SUM(salary) AS total_salary
FROM employees
GROUP BY dept_id;

-- sub query goes into the below query
SELECT dept_id, total_salary
FROM (
    SELECT dept_id, SUM(salary) AS total_salary
    FROM employees
    GROUP BY dept_id
) AS dept_salaries
WHERE total_salary > 120000;

-- ==================================
-- Subquery with IN (multiple values)
-- ==================================

-- subquery 
SELECT dept_id
FROM employees
GROUP BY dept_id
HAVING COUNT(employee_id) > 2;

SELECT first_name, last_name, dept_id
FROM employees
WHERE dept_id IN (
    SELECT dept_id
    FROM employees
    GROUP BY dept_id
    HAVING COUNT(employee_id) > 2
);


-- ============================================
-- Show departments where total salary > 70000.
-- ==========================================
SELECT dept_id, SUM(salary) AS total_salary
FROM employees
GROUP BY dept_id
HAVING SUM(salary) > 120000;

-- ==========
-- Group by 
-- ==========
# Group employees by department and calculate total salary per department.
SELECT dept_id, SUM(salary) AS total_salary
FROM employees
GROUP BY dept_id;

# Count employees and calculate average salary per department.
SELECT dept_id, COUNT(employee_id) AS employee_count, AVG(salary) AS avg_salary
FROM employees
GROUP BY dept_id;

#  GROUP BY with Multiple Columns
SELECT dept_id,
       CASE 
           WHEN salary >= 70000 THEN 'High'
           WHEN salary >= 60000 THEN 'Medium'
           ELSE 'Low'
       END AS salary_level,
       COUNT(*) AS emp_count
FROM employees
GROUP BY dept_id, salary_level;

-- =============================
-- Indexes 
-- =============================

-- Check Indexes on a Table
SHOW INDEXES FROM employees;

-- Create an index
CREATE INDEX idx_last_name ON employees(last_name);
select * from employees;

-- Search employees by last name
SELECT * 
FROM employees
WHERE last_name = 'Lee';

-- Creating a Composite Index
CREATE INDEX idx_dept_salary ON employees(dept_id, salary);
SELECT * 
FROM employees
WHERE dept_id = 2 AND salary > 70000;

-- Drop an Index-- 
DROP INDEX idx_last_name ON employees;



