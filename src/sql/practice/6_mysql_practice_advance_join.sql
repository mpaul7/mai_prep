 USE db_1;
show tables;
select * from staff;
select * from contractors;
-- ===================================
-- Advanced Joins and Set Operations 
-- ====================================

CREATE TABLE staff (
    staff_id INT PRIMARY KEY,
    staff_name VARCHAR(50),
    department VARCHAR(50)
);

CREATE TABLE contractors (
    contractor_id INT PRIMARY KEY,
    contractor_name VARCHAR(50),
    department VARCHAR(50)
);

INSERT INTO staff (staff_id, staff_name, department) VALUES
(1, 'Alice', 'HR'),
(2, 'Bob', 'IT'),
(3, 'Charlie', 'Finance');

INSERT INTO contractors (contractor_id, contractor_name, department) VALUES
(101, 'David', 'IT'),
(102, 'Eve', 'Finance'),
(103, 'Frank', 'Marketing');


select * from staff;
select * from contractors;

-- ==============================================================================
-- Write a query that uses a UNION to combine results from two different queries.
-- ==============================================================================
  -- Combine all staff and contractors into a single list:
  -- it is line concat df = pd.concat([df1, df2])
SELECT staff_name AS name, department
FROM staff

UNION

SELECT contractor_name AS name, department
FROM contractors;

-- CROSS JOIN Example
CREATE TABLE team_departments (
    team_name VARCHAR(50)
);
INSERT INTO team_departments VALUES ('HR'), ('IT'), ('Finance'), ('Marketing');
select * from team_departments;

SELECT s.staff_name, t.team_name
FROM staff s
CROSS JOIN team_departments t;

-- INTERSECT (MySQL-Compatible Version)
-- it is like select those rows from DF were column values in some list in []
-- columns = ['a', 'b', 'c']
-- df = df[df['col'] in columns]
SELECT staff_name AS name
FROM staff
WHERE staff_name IN (SELECT contractor_name FROM contractors);


-- EXCEPT (MySQL-Compatible Version)
SELECT staff_name AS name
FROM staff
WHERE staff_name NOT IN (SELECT contractor_name FROM contractors);

