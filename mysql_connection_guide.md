# MySQL Connection Guide

## Current Issue
Error: Access denied for user 'mpaul'@'localhost' (using password: YES)

## Solution Steps

### 1. Connect as MySQL root user
```bash
sudo mysql -u root
```

### 2. Create proper user and database
```sql
-- Create user with password
CREATE USER IF NOT EXISTS 'analyst'@'localhost' IDENTIFIED BY 'practice123';

-- Create practice database
CREATE DATABASE IF NOT EXISTS practice_db;

-- Grant privileges
GRANT ALL PRIVILEGES ON practice_db.* TO 'analyst'@'localhost';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, INDEX, ALTER ON *.* TO 'analyst'@'localhost';

-- Apply changes
FLUSH PRIVILEGES;

-- Verify user was created
SELECT user, host FROM mysql.user WHERE user = 'analyst';

-- Exit
EXIT;
```

### 3. Test connection
```bash
mysql -u analyst -p practice_db
```
Enter password: practice123

### 4. Cursor Extension Connection Settings
- Host: localhost
- Port: 3306
- Username: analyst
- Password: practice123
- Database: practice_db

## Alternative: Use root user directly
If you prefer to use root user in Cursor:
- Username: root
- Password: (leave empty for socket authentication, or set a password)
- Database: practice_db

## Troubleshooting
If still getting access denied:
1. Check MySQL authentication method:
   ```sql
   SELECT user, host, plugin FROM mysql.user;
   ```

2. For Ubuntu/Debian, root might use auth_socket plugin
3. Create a new user specifically for external connections
