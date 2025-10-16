
show tables;

-- ===============
-- create tables 
-- ===============
CREATE TABLE accounts (
    account_id INT PRIMARY KEY AUTO_INCREMENT,
    account_name VARCHAR(50),
    balance DECIMAL(10,2)
);

CREATE TABLE transactions (
    txn_id INT PRIMARY KEY AUTO_INCREMENT,
    account_id INT,
    amount DECIMAL(10,2),
    txn_type VARCHAR(10),
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

-- ======================
-- Start a transaction
-- ======================
select * from accounts;
select * from transactions;

START TRANSACTION;

-- Step 1: Insert into parent table (accounts)
INSERT INTO accounts (account_name, balance)
VALUES ('Alice', 1000.00);

-- Step 2: Get the last inserted account_id
SET @acc_id = LAST_INSERT_ID();

-- Step 3: Insert into child table (transactions)
INSERT INTO transactions (account_id, amount, txn_type)
VALUES (@acc_id, 100.00, 'DEBIT');
INSERT INTO transactions (account_id, amount, txn_type)
VALUES (@acc_id, 200.00, 'CREDIT');

-- Step 4: Commit the transaction
COMMIT;


START TRANSACTION;

INSERT INTO accounts (account_name, balance)
VALUES ('Bob', 500.00);

-- Intentional error: Invalid account_id reference
INSERT INTO transactions (account_id, amount, txn_type)
VALUES (99999, 50.00, 'CREDIT');

-- Rollback the entire transaction if any error occurs
ROLLBACK;