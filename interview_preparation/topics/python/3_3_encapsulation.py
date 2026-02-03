"""
Python Encapsulation - Interview Preparation
Topic 3.3: Encapsulation

This module covers:
- Access Modifiers: Public, protected (_), private (__)
- Properties: @property decorator, getters, setters
"""

# ============================================================================
# 1. ACCESS MODIFIERS
# ============================================================================

print("=" * 70)
print("1. ACCESS MODIFIERS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Public Attributes
# ----------------------------------------------------------------------------
print("\n--- 1.1 Public Attributes ---")

class Person:
    """Person class with public attributes."""
    
    def __init__(self, name, age):
        # Public attributes - accessible from anywhere
        self.name = name
        self.age = age

person = Person("Alice", 25)
print(f"Name: {person.name}")  # Direct access
print(f"Age: {person.age}")    # Direct access

# Can modify directly
person.name = "Bob"
person.age = 30
print(f"After modification: {person.name}, {person.age}")


# ----------------------------------------------------------------------------
# 1.2 Protected Attributes (Single Underscore)
# ----------------------------------------------------------------------------
print("\n--- 1.2 Protected Attributes (Single Underscore) ---")

class Person:
    """Person class with protected attributes."""
    
    def __init__(self, name, age):
        self.name = name
        self._age = age  # Protected attribute (convention)
    
    def get_age(self):
        """Getter method for protected attribute."""
        return self._age
    
    def set_age(self, age):
        """Setter method with validation."""
        if age < 0:
            raise ValueError("Age cannot be negative")
        self._age = age

person = Person("Alice", 25)

# Protected attributes can still be accessed (Python doesn't enforce)
print(f"Name: {person.name}")
print(f"Age (direct): {person._age}")  # Works but not recommended
print(f"Age (getter): {person.get_age()}")  # Recommended way

# Convention: Single underscore means "internal use"
# Not enforced by Python, but indicates "don't access directly"


# ----------------------------------------------------------------------------
# 1.3 Private Attributes (Double Underscore)
# ----------------------------------------------------------------------------
print("\n--- 1.3 Private Attributes (Double Underscore) ---")

class Person:
    """Person class with private attributes."""
    
    def __init__(self, name, age, ssn):
        self.name = name
        self._age = age  # Protected
        self.__ssn = ssn  # Private (name mangling)
    
    def get_ssn(self):
        """Getter for private attribute."""
        return self.__ssn
    
    def set_ssn(self, ssn):
        """Setter for private attribute."""
        self.__ssn = ssn

person = Person("Alice", 25, "123-45-6789")

print(f"Name: {person.name}")
print(f"Age: {person._age}")
print(f"SSN (getter): {person.get_ssn()}")

# Cannot access directly (name mangling)
# print(person.__ssn)  # Would raise AttributeError

# Name mangling: __ssn becomes _Person__ssn
print(f"SSN (mangled): {person._Person__ssn}")  # Works but not recommended!

# Private attributes are "name mangled" to _ClassName__attribute
# This prevents accidental access but can still be accessed if needed


# ----------------------------------------------------------------------------
# 1.4 Understanding Name Mangling
# ----------------------------------------------------------------------------
print("\n--- 1.4 Understanding Name Mangling ---")

class BankAccount:
    """Bank account with private balance."""
    
    def __init__(self, balance):
        self.__balance = balance  # Private attribute
    
    def get_balance(self):
        """Getter for balance."""
        return self.__balance
    
    def deposit(self, amount):
        """Deposit money."""
        self.__balance += amount
    
    def withdraw(self, amount):
        """Withdraw money."""
        if amount <= self.__balance:
            self.__balance -= amount
            return True
        return False

account = BankAccount(1000)
print(f"Balance (getter): ${account.get_balance()}")

# Name mangling demonstration
print(f"Balance (mangled name): ${account._BankAccount__balance}")

# Direct access fails
try:
    print(account.__balance)
except AttributeError as e:
    print(f"Cannot access directly: {e}")


# ----------------------------------------------------------------------------
# 1.5 Protected vs Private Methods
# ----------------------------------------------------------------------------
print("\n--- 1.5 Protected vs Private Methods ---")

class Calculator:
    """Calculator with protected and private methods."""
    
    def __init__(self):
        self._history = []  # Protected attribute
    
    def add(self, a, b):
        """Public method."""
        result = self._calculate(a, b, '+')  # Call protected method
        return result
    
    def _calculate(self, a, b, operation):
        """Protected method - internal use."""
        self._log_operation(a, b, operation)  # Call private method
        if operation == '+':
            return a + b
        return None
    
    def __log_operation(self, a, b, operation):
        """Private method - name mangled."""
        self._history.append(f"{a} {operation} {b}")

calc = Calculator()
result = calc.add(5, 3)
print(f"5 + 3 = {result}")

# Can access protected method (not recommended)
# calc._calculate(5, 3, '+')

# Cannot access private method directly
# calc.__log_operation(5, 3, '+')  # Would raise AttributeError

# Can access via mangled name (not recommended)
# calc._Calculator__log_operation(5, 3, '+')


# ============================================================================
# 2. PROPERTIES (@property)
# ============================================================================

print("\n" + "=" * 70)
print("2. PROPERTIES (@property)")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Basic Property (Read-Only)
# ----------------------------------------------------------------------------
print("\n--- 2.1 Basic Property (Read-Only) ---")

class Circle:
    """Circle with property for area."""
    
    def __init__(self, radius):
        self._radius = radius  # Protected attribute
    
    @property
    def radius(self):
        """Getter for radius."""
        return self._radius
    
    @property
    def area(self):
        """Read-only property for area."""
        import math
        return math.pi * self._radius ** 2
    
    @property
    def diameter(self):
        """Read-only property for diameter."""
        return 2 * self._radius

circle = Circle(5)
print(f"Radius: {circle.radius}")
print(f"Area: {circle.area:.2f}")  # Access like attribute
print(f"Diameter: {circle.diameter}")

# Cannot set read-only property
# circle.area = 100  # Would raise AttributeError


# ----------------------------------------------------------------------------
# 2.2 Property with Setter
# ----------------------------------------------------------------------------
print("\n--- 2.2 Property with Setter ---")

class Temperature:
    """Temperature class with property and setter."""
    
    def __init__(self, celsius):
        self._celsius = celsius  # Store in Celsius
    
    @property
    def celsius(self):
        """Getter for Celsius."""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Setter for Celsius with validation."""
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        """Getter for Fahrenheit."""
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        """Setter for Fahrenheit."""
        self.celsius = (value - 32) * 5/9

temp = Temperature(25)
print(f"Celsius: {temp.celsius}°C")
print(f"Fahrenheit: {temp.fahrenheit:.1f}°F")

# Set via property
temp.celsius = 30
print(f"\nAfter setting to 30°C: {temp.fahrenheit:.1f}°F")

temp.fahrenheit = 100
print(f"After setting to 100°F: {temp.celsius:.1f}°C")

# Validation works
try:
    temp.celsius = -300
except ValueError as e:
    print(f"\nValidation error: {e}")


# ----------------------------------------------------------------------------
# 2.3 Property with Getter, Setter, and Deleter
# ----------------------------------------------------------------------------
print("\n--- 2.3 Property with Getter, Setter, and Deleter ---")

class Person:
    """Person class with property, setter, and deleter."""
    
    def __init__(self, name):
        self._name = name
    
    @property
    def name(self):
        """Getter for name."""
        print("Getting name")
        return self._name
    
    @name.setter
    def name(self, value):
        """Setter for name with validation."""
        print(f"Setting name to {value}")
        if not isinstance(value, str):
            raise TypeError("Name must be a string")
        if len(value) == 0:
            raise ValueError("Name cannot be empty")
        self._name = value
    
    @name.deleter
    def name(self):
        """Deleter for name."""
        print("Deleting name")
        self._name = "Unknown"

person = Person("Alice")
print(f"Name: {person.name}")

person.name = "Bob"
print(f"Name: {person.name}")

del person.name
print(f"Name after delete: {person.name}")


# ----------------------------------------------------------------------------
# 2.4 Property vs Regular Methods
# ----------------------------------------------------------------------------
print("\n--- 2.4 Property vs Regular Methods ---")

class Rectangle:
    """Rectangle comparing property vs methods."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    # Using property (accessed like attribute)
    @property
    def area(self):
        return self.width * self.height
    
    # Using method (must call with parentheses)
    def get_perimeter(self):
        return 2 * (self.width + self.height)

rect = Rectangle(5, 3)

# Property: accessed like attribute
print(f"Area: {rect.area}")  # No parentheses

# Method: must call with parentheses
print(f"Perimeter: {rect.get_perimeter()}")  # With parentheses

# Properties provide cleaner syntax for computed attributes


# ----------------------------------------------------------------------------
# 2.5 Property for Validation
# ----------------------------------------------------------------------------
print("\n--- 2.5 Property for Validation ---")

class BankAccount:
    """Bank account with validated balance property."""
    
    def __init__(self, balance):
        self._balance = balance
    
    @property
    def balance(self):
        """Getter for balance."""
        return self._balance
    
    @balance.setter
    def balance(self, value):
        """Setter with validation."""
        if value < 0:
            raise ValueError("Balance cannot be negative")
        self._balance = value
    
    def withdraw(self, amount):
        """Withdraw money."""
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self.balance = self._balance - amount  # Uses setter (validates)

account = BankAccount(1000)
print(f"Initial balance: ${account.balance}")

account.balance = 1500  # Uses setter
print(f"After deposit: ${account.balance}")

account.withdraw(200)
print(f"After withdrawal: ${account.balance}")

# Validation prevents negative balance
try:
    account.balance = -100
except ValueError as e:
    print(f"\nValidation error: {e}")


# ----------------------------------------------------------------------------
# 2.6 Property for Computed Attributes
# ----------------------------------------------------------------------------
print("\n--- 2.6 Property for Computed Attributes ---")

class Student:
    """Student with computed GPA property."""
    
    def __init__(self, name):
        self.name = name
        self._grades = []
    
    def add_grade(self, grade):
        """Add a grade."""
        if 0 <= grade <= 100:
            self._grades.append(grade)
        else:
            raise ValueError("Grade must be between 0 and 100")
    
    @property
    def gpa(self):
        """Computed property for GPA."""
        if not self._grades:
            return 0.0
        return sum(self._grades) / len(self._grades)
    
    @property
    def grade_count(self):
        """Number of grades."""
        return len(self._grades)

student = Student("Alice")
student.add_grade(85)
student.add_grade(90)
student.add_grade(78)

print(f"Student: {student.name}")
print(f"Grades: {student._grades}")
print(f"GPA: {student.gpa:.2f}")  # Computed property
print(f"Number of grades: {student.grade_count}")

# GPA updates automatically when grades change
student.add_grade(95)
print(f"\nAfter adding grade: GPA = {student.gpa:.2f}")


# ============================================================================
# 3. PRACTICAL EXAMPLES
# ============================================================================

print("\n" + "=" * 70)
print("3. PRACTICAL EXAMPLES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Complete Example: Employee Class
# ----------------------------------------------------------------------------
print("\n--- 3.1 Complete Example: Employee Class ---")

class Employee:
    """Employee class demonstrating encapsulation."""
    
    # Class variable
    company = "Tech Corp"
    
    def __init__(self, name, employee_id, salary):
        # Public attributes
        self.name = name
        self.employee_id = employee_id
        
        # Protected attribute
        self._salary = salary
        
        # Private attribute
        self.__bonus = 0
    
    @property
    def salary(self):
        """Getter for salary."""
        return self._salary
    
    @salary.setter
    def salary(self, value):
        """Setter for salary with validation."""
        if value < 0:
            raise ValueError("Salary cannot be negative")
        self._salary = value
    
    def get_bonus(self):
        """Getter for private bonus."""
        return self.__bonus
    
    def set_bonus(self, amount):
        """Setter for private bonus."""
        if amount < 0:
            raise ValueError("Bonus cannot be negative")
        self.__bonus = amount
    
    def get_total_compensation(self):
        """Calculate total compensation."""
        return self._salary + self.__bonus

employee = Employee("Alice", "E001", 50000)
print(f"Employee: {employee.name}")
print(f"Salary: ${employee.salary}")

employee.salary = 55000  # Uses property setter
print(f"New salary: ${employee.salary}")

employee.set_bonus(5000)
print(f"Bonus: ${employee.get_bonus()}")
print(f"Total compensation: ${employee.get_total_compensation()}")


# ----------------------------------------------------------------------------
# 3.2 Example: Rectangle with Properties
# ----------------------------------------------------------------------------
print("\n--- 3.2 Example: Rectangle with Properties ---")

class Rectangle:
    """Rectangle with width and height properties."""
    
    def __init__(self, width, height):
        self._width = width
        self._height = height
    
    @property
    def width(self):
        return self._width
    
    @width.setter
    def width(self, value):
        if value <= 0:
            raise ValueError("Width must be positive")
        self._width = value
    
    @property
    def height(self):
        return self._height
    
    @height.setter
    def height(self, value):
        if value <= 0:
            raise ValueError("Height must be positive")
        self._height = value
    
    @property
    def area(self):
        """Computed property for area."""
        return self._width * self._height
    
    @property
    def perimeter(self):
        """Computed property for perimeter."""
        return 2 * (self._width + self._height)

rect = Rectangle(5, 3)
print(f"Width: {rect.width}, Height: {rect.height}")
print(f"Area: {rect.area}")
print(f"Perimeter: {rect.perimeter}")

rect.width = 6  # Uses setter
print(f"\nAfter changing width: Area = {rect.area}")


# ============================================================================
# 4. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("4. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. ACCESS MODIFIERS:
   - Public: No underscore (default)
   - Protected: Single underscore (_) - convention, not enforced
   - Private: Double underscore (__) - name mangling
   - Python doesn't enforce access control, uses conventions

2. PROTECTED ATTRIBUTES (_):
   - Single underscore prefix
   - Convention: "internal use"
   - Can still be accessed directly
   - Indicates "don't access from outside class"
   - Used for attributes that subclasses might need

3. PRIVATE ATTRIBUTES (__):
   - Double underscore prefix
   - Name mangling: __attr becomes _ClassName__attr
   - Prevents accidental access
   - Can still be accessed via mangled name (not recommended)
   - Used for truly internal implementation

4. PROPERTIES (@property):
   - Allows computed attributes
   - Accessed like attributes (no parentheses)
   - Can have getter, setter, deleter
   - Provides validation and encapsulation
   - Cleaner syntax than getter/setter methods

5. PROPERTY DECORATORS:
   - @property: Defines getter
   - @property_name.setter: Defines setter
   - @property_name.deleter: Defines deleter
   - All use same method name

6. WHEN TO USE PROPERTIES:
   - Computed attributes (area, perimeter)
   - Validation needed (salary, age)
   - Lazy evaluation
   - Maintaining backward compatibility
   - When attribute access is more natural than method calls

7. PROPERTY VS METHODS:
   - Property: No parentheses, computed value
   - Method: Parentheses required, can take parameters
   - Use property when value is attribute-like
   - Use method when action/operation is performed

8. BEST PRACTICES:
   - Use protected (_) for internal attributes
   - Use private (__) sparingly (only when truly needed)
   - Use properties for computed/validated attributes
   - Document access conventions
   - Don't rely on name mangling for security

9. INTERVIEW TIPS:
   - Know difference between _, __, and no underscore
   - Understand name mangling mechanism
   - Know how to use @property decorator
   - Understand when to use properties vs methods
   - Be able to implement getters/setters with properties
   - Explain Python's philosophy (convention over enforcement)
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Encapsulation Guide Ready!")
    print("=" * 70)
