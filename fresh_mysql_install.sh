#!/bin/bash

# Fresh MySQL Installation Script
# Run this script to completely reinstall MySQL

echo "=== MySQL Fresh Installation Script ==="
echo "This will completely remove and reinstall MySQL"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

echo "Step 1: Stopping MySQL service..."
sudo systemctl stop mysql

echo "Step 2: Removing MySQL packages..."
sudo apt-get remove --purge mysql-server mysql-client mysql-common mysql-server-core-* mysql-client-core-* -y

echo "Step 3: Removing MySQL data and config directories..."
sudo rm -rf /var/lib/mysql
sudo rm -rf /var/log/mysql
sudo rm -rf /etc/mysql

echo "Step 4: Cleaning up..."
sudo apt-get autoremove -y
sudo apt-get autoclean

echo "Step 5: Updating package list..."
sudo apt-get update

# echo "Step 6: Installing MySQL server..."
# sudo apt-get install mysql-server -y

# echo "Step 7: Starting MySQL service..."
# sudo systemctl start mysql
# sudo systemctl enable mysql

# echo "Step 8: Checking MySQL status..."
# sudo systemctl status mysql

# echo ""
# echo "=== Installation Complete! ==="
# echo "Next steps:"
# echo "1. Run: sudo mysql_secure_installation (optional)"
# echo "2. Connect as root: sudo mysql -u root"
# echo "3. Create database and user as shown in the guide"
# echo ""
