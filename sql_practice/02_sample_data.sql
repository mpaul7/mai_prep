-- Sample Data for Practice
-- Insert sample data into tables

-- Insert Departments
INSERT INTO departments (department_name, location) VALUES
('Sales', 'New York'),
('Marketing', 'Los Angeles'),
('IT', 'San Francisco'),
('HR', 'Chicago'),
('Finance', 'Boston');

-- Insert Employees
INSERT INTO employees (first_name, last_name, email, department, salary, hire_date, manager_id) VALUES
('John', 'Smith', 'john.smith@company.com', 'Sales', 75000.00, '2022-01-15', NULL),
('Sarah', 'Johnson', 'sarah.johnson@company.com', 'Marketing', 68000.00, '2022-02-20', NULL),
('Mike', 'Davis', 'mike.davis@company.com', 'IT', 85000.00, '2021-11-10', NULL),
('Emily', 'Brown', 'emily.brown@company.com', 'Sales', 62000.00, '2022-03-05', 1),
('David', 'Wilson', 'david.wilson@company.com', 'IT', 78000.00, '2022-01-30', 3),
('Lisa', 'Anderson', 'lisa.anderson@company.com', 'HR', 71000.00, '2021-12-15', NULL),
('Tom', 'Taylor', 'tom.taylor@company.com', 'Finance', 73000.00, '2022-02-10', NULL),
('Anna', 'Martinez', 'anna.martinez@company.com', 'Marketing', 65000.00, '2022-04-01', 2);

-- Insert Customers
INSERT INTO customers (first_name, last_name, email, city, country, registration_date) VALUES
('Alice', 'Cooper', 'alice.cooper@email.com', 'New York', 'USA', '2022-01-10'),
('Bob', 'Miller', 'bob.miller@email.com', 'Los Angeles', 'USA', '2022-01-15'),
('Carol', 'Garcia', 'carol.garcia@email.com', 'Chicago', 'USA', '2022-02-01'),
('Daniel', 'Rodriguez', 'daniel.rodriguez@email.com', 'Houston', 'USA', '2022-02-15'),
('Eva', 'Lopez', 'eva.lopez@email.com', 'Phoenix', 'USA', '2022-03-01'),
('Frank', 'Lee', 'frank.lee@email.com', 'Philadelphia', 'USA', '2022-03-10'),
('Grace', 'Wang', 'grace.wang@email.com', 'San Antonio', 'USA', '2022-03-20'),
('Henry', 'Kim', 'henry.kim@email.com', 'San Diego', 'USA', '2022-04-01');

-- Insert Sales
INSERT INTO sales (employee_id, product_name, sale_amount, sale_date, customer_id) VALUES
(1, 'Laptop Pro', 1299.99, '2022-03-15', 1),
(1, 'Wireless Mouse', 29.99, '2022-03-16', 2),
(4, 'Monitor 4K', 399.99, '2022-03-20', 3),
(1, 'Keyboard Mechanical', 149.99, '2022-03-25', 4),
(4, 'Webcam HD', 89.99, '2022-04-01', 5),
(1, 'Headphones', 199.99, '2022-04-05', 6),
(4, 'Tablet', 599.99, '2022-04-10', 7),
(1, 'Smartphone', 899.99, '2022-04-15', 8),
(4, 'Smartwatch', 299.99, '2022-04-20', 1),
(1, 'Bluetooth Speaker', 79.99, '2022-04-25', 2);
