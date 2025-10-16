
show tables;
SELECT USER(), CURRENT_USER();
select * from employees;

drop table if exists employees;
drop table if exists departments;

-- ================================
-- Data Definition Language (DDL)
-- ===============================
CREATE TABLE employees (
    employee_id INT AUTO_INCREMENT PRIMARY KEY,   -- Integer + primary key
    first_name VARCHAR(50) NOT NULL,             -- String, not null
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,                   -- Unique email
    hire_date DATE,                              -- Date
    salary DECIMAL(10,2),                        -- Decimal for salary
    department VARCHAR(50)                        -- String for department
);

CREATE TABLE employees (
    employee_id INT AUTO_INCREMENT PRIMARY KEY,   -- Integer + primary key
    first_name VARCHAR(50) NOT NULL,             -- String, not null
    last_name VARCHAR(50) NOT NULL,
    salary DECIMAL(10,2),                        -- Decimal for salary
    department VARCHAR(50)                       -- String for department
);


-- Alter table 
-- =========================

alter table employees 
add column manager_name varchar(50);

alter table employees 
add column dept_id varchar(50);

alter table employees 
modify column salary decimal(12, 2);

alter table employees;
SELECT DATABASE();
rename column department to dept_name;

alter table employees drop column manager_name;

-- ========================================
-- Data Manpulation language (DML)
-- ========================================
select * from employees;

INSERT INTO employees 
(first_name, last_name, salary, department) 
VALUES 
('Alice', 'Johnson', 60000, 'HR'),
('Alex', 'Chan', 62000, 'HR');

INSERT INTO employees 
(first_name, last_name, salary, department)
VALUES 
('Bob', 'Smith', 75000, 'Engineering'),
('Bib', 'Smyth', 78000, 'Engineering'),
('Charlie', 'Lee', 50000, 'Marketing'),
('Chad', 'Lee', 55000, 'Marketing'),
('Diana', 'Brown', 70000, 'Sales'),
('Dan', 'Brew', 75000, 'Sales');

INSERT INTO employees 
(dept_id)
VALUES 
(1, 2, 3, 4);
-- ============================
-- Update records
-- ============================

UPDATE employees 
SET salary = salary * 1.10 
WHERE dept_name = 'Engineering';

UPDATE employees SET salary = salary + 1000;
