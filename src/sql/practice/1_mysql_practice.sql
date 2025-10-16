USE db_1;
show tables;
drop table if exists employees;
drop table if exists departments;
drop table if exists accounts;
drop table if exists contractors;
drop table if exists staff;
drop table if exists team_departments;
drop table if exists transactions;
drop table if exists vw_high_salary;

-- ======================================
-- Create department table 

create table departments (
	dept_id int auto_increment primary key,
    dept_name varchar(50)
);

 --   create employee table
 CREATE TABLE employees (
    employee_id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100),
    salary DECIMAL(10,2),
    dept_id INT,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);
 -- ======================================
 -- insert data into  tables
--  =======================================
insert into departments (dept_name) values ('HR'), ('Engineering'), ('Marketing'), ('Sales');
INSERT INTO employees (first_name, last_name, email, salary, dept_id) VALUES
('Alice', 'Johnson', 'alice.johnson@example.com', 60000, 1),
('Bob', 'Smith', 'bob.smith@example.com', 75000, 2),
('Charlie', 'Lee', 'charlie.lee@example.com', 50000, 3),
('Diana', 'Brown', 'diana.brown@example.com', 70000, NULL);

--  =========================================
--  Check data in tables
--  =========================================

select * from employees;
select * from departments;
-- ===========================================
-- 7. Update and Delete
-- ===========================================
update employees 
set email = 'bob.smith2025@gmail.com' 
where first_name = 'Bob' and last_name = 'Smith';

delete from employees
where first_name = 'charlie'

-- ===========================================
-- 8. Aggregate Functions
-- ===========================================
-- total employees
select * from employees
select salary from employees;

select count(*) as total_employees from employees; 
select sum(salary) as total_salary from employees;
select avg(salary) as average_salary from employees;
select min(salary) as min_salary from employees;
select max(salary) as max_salary from employees;
SELECT MIN(salary) AS min_salary, MAX(salary) AS max_salary FROM employees;
-- ===========================================
-- 8. Join tables
-- ===========================================
-- Left join

select 
e.first_name,
e.last_name,
d.dept_id,
d.dept_name
from employees as e
left join departments as d 
on e.dept_id = d.dept_id;

select 
e.employee_id, 
e.first_name,
e.last_name,
d.dept_id,
d.dept_name
from departments as d
left join employees as e
on d.dept_id = e.dept_id;

-- inner join
select 
e.first_name,
e.last_name,
d.dept_id,
d.dept_name
from employees as e
inner join departments as d 
on e.dept_id = d.dept_id;

-- right join
select 
e.first_name,
e.last_name,
d.dept_id
from employees as e
right join departments as d 
on e.dept_id = d.dept_id;

-- full join
select *
from employees as e
left join departments as d 
on e.dept_id = d.dept_id

union 

select *
from employees as e
right join departments as d 
on e.dept_id = d.dept_id;

-- ===========================================
-- Aggregate using group by
-- ===========================================
-- count employees per department
select d.dept_name, count(e.employee_id) as employee_count
from departments d
left join employees e 
on d.dept_id = e.dept_id
group by d.dept_name;

-- average salaary per department

select d.dept_name, 
avg(e.salary) as average_salary
from departments d 
left join employees e 
on d.dept_id = e.dept_id
group by d.dept_name;

select d.dept_name, 
avg(e.salary) as average_salary, count(e.employee_id) as employee_count
from departments d 
left join employees e 
on d.dept_id = e.dept_id
group by d.dept_name;

-- ===========================================
-- 9. Sorting and Limiting
-- ===========================================

-- employees sorted by salary descending 
select first_name, last_name, salary
from employees
order by salary desc;

-- Top 2 highest paid employees
select first_name, last_name, salary 
from employees
order by salary desc
limit 2;

-- Top 2 least paid employees
select first_name, last_name, salary 
from employees
order by salary asc
limit 2;

-- top 10 employes with 10% bonus
SELECT first_name, last_name, salary, salary * 1.10 AS salary_with_bonus 
FROM employees 
ORDER BY salary DESC 
LIMIT 5;

-- Salary difference from avearage 
SELECT first_name, last_name, salary, salary - (SELECT AVG(salary)FROM employees) AS diff_from_avg 
FROM employees;