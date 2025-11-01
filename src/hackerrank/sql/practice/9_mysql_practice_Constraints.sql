show tables;
drop table if exists employees;
drop table if exists departments;

select * from employees;
select * from departments;

CREATE TABLE departments (
    dept_id INT PRIMARY KEY,
    dept_name VARCHAR(50) UNIQUE NOT NULL,
    location VARCHAR(50) CHECK (location IN ('NY', 'LA', 'TX'))
);

CREATE TABLE employees (
    emp_id INT PRIMARY KEY,
    emp_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    salary DECIMAL(10,2) CHECK (salary > 0),
    dept_id INT,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);

INSERT INTO departments VALUES
(1, 'HR', 'NY'),
(2, 'IT', 'LA'),
(3, 'Finance', 'TX');

INSERT INTO employees VALUES
(101, 'Alice', 'alice@example.com', 5000, 1),
(102, 'Bob', 'bob@example.com', 7000, 2),
(103, 'Charlie', 'charlie@example.com', 6000, 3);

-- ==============================
-- PRIMARY KEY Violation
INSERT INTO employees VALUES
(101, 'David', 'david@example.com', 4000, 2);

-- FOREIGN KEY Violation
INSERT INTO employees VALUES
(104, 'Eve', 'eve@example.com', 5500, 9);

-- UNIQUE Violation
INSERT INTO employees VALUES
(105, 'Frank', 'alice@example.com', 8000, 2);

-- CHECK Violation
INSERT INTO employees VALUES
(106, 'Grace', 'grace@example.com', -3000, 3);

-- NOT NULL Violation
INSERT INTO employees VALUES
(107, NULL, 'no_name@example.com', 5000, 1);





