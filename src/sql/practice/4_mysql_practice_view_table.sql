show tables;
-- Listing Views in Database
SHOW FULL TABLES IN db_1 WHERE TABLE_TYPE LIKE 'VIEW';

drop table if exists employees;
drop table if exists departments;


create table departments (
	dept_id int auto_increment primary key,
    dept_name varchar(50)
);

CREATE TABLE employees (
    employee_id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100),
    salary DECIMAL(10,2),
    dept_id INT,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);

insert into departments (dept_name) values ('HR'), ('Engineering'), ('Marketing'), ('Sales');
INSERT INTO employees (first_name, last_name, email, salary, dept_id) VALUES
('Alice', 'Johnson', 'alice.johnson@example.com', 60000, 1),
('Bob', 'Smith', 'bob.smith@example.com', 75000, 2),
('Charlie', 'Lee', 'charlie.lee@example.com', 50000, 3),
('Diana', 'Brown', 'diana.brown@example.com', 70000, NULL);


select * from employees;
select * from departments;
-- =========================
--  Creating a Basic View 
-- =========================
-- Is like sub of dataFrame 
-- df2 = df[df['col1', 'col2']]
CREATE VIEW vw_employee_details AS
SELECT e.employee_id,
       e.first_name,
       e.last_name,
       e.salary,
       d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id;

SELECT * FROM vw_employee_details;

SELECT dept_name, AVG(salary) AS avg_salary
FROM vw_employee_details
GROUP BY dept_name;


-- Create a view that only shows employees earning above 60,000.
CREATE  VIEW vw_high_salary AS
SELECT first_name, last_name, salary, dept_id
FROM employees
WHERE salary > 60000;

SELECT * FROM vw_high_salary;

-- Updating Data Through a View (When Allowed)-- 
UPDATE vw_high_salary
SET salary = salary + 2000
WHERE first_name = 'Bob';

-- Dropping a View
DROP VIEW IF EXISTS vw_employee_details;



