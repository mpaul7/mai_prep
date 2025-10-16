-- =============================
-- Common Table Expression (CTE)
-- =============================

show tables;
select * from employees;


-- The inner query (CTE) calculates the average once.
-- The outer query references it easily, improving readability.
with avg_salary as (
	select avg(salary) as avg_sal
    from employees
)
select e.employee_id, e.first_name, e.salary
from employees e, avg_salary a
where e.salary > a.avg_sal;

-- =============================
-- CTE with Multiple CTEs
-- =============================

with dept_total as (
	select dept_id, sum(salary) as total_sal
    from employees
    group by dept_id
),
high_salary as (
   select * from dept_total where total_sal > 60000
)
select d.dept_id, d.total_sal
from high_salary d;