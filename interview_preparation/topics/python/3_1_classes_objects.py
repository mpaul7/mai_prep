"""
Python Classes & Objects - Interview Preparation
Topic 3.1: Classes & Objects

This module covers:
- Class Definition: class keyword, __init__
- Instance Variables: self, instance attributes
- Class Variables: Class attributes vs instance attributes
- Methods: Instance methods, class methods (@classmethod), static methods (@staticmethod)
- Magic Methods: __str__, __repr__, __len__, __eq__, __lt__, __hash__
"""

# ============================================================================
# 1. CLASS DEFINITION
# ============================================================================

print("=" * 70)
print("1. CLASS DEFINITION")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Basic Class Definition
# ----------------------------------------------------------------------------
print("\n--- 1.1 Basic Class Definition ---")

class Person:
    """Simple Person class."""
    pass

# Create instance (object)
person1 = Person()
print(f"Person instance: {person1}")
print(f"Type: {type(person1)}")


# ----------------------------------------------------------------------------
# 1.2 Class with __init__ Method
# ----------------------------------------------------------------------------
print("\n--- 1.2 Class with __init__ Method ---")

class Person:
    """Person class with __init__ constructor."""
    
    def __init__(self, name, age):
        """Initialize person with name and age."""
        self.name = name
        self.age = age

# Create instances
person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

print(f"Person 1: {person1.name}, {person1.age} years old")
print(f"Person 2: {person2.name}, {person2.age} years old")


# ----------------------------------------------------------------------------
# 1.3 Class with Default Parameters
# ----------------------------------------------------------------------------
print("\n--- 1.3 Class with Default Parameters ---")

class Person:
    """Person class with default parameters."""
    
    def __init__(self, name="Unknown", age=0):
        """Initialize person with optional name and age."""
        self.name = name
        self.age = age

# Create instances with different arguments
person1 = Person("Alice", 25)
person2 = Person("Bob")  # age defaults to 0
person3 = Person()  # both default

print(f"Person 1: {person1.name}, {person1.age}")
print(f"Person 2: {person2.name}, {person2.age}")
print(f"Person 3: {person3.name}, {person3.age}")


# ============================================================================
# 2. INSTANCE VARIABLES
# ============================================================================

print("\n" + "=" * 70)
print("2. INSTANCE VARIABLES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Understanding self
# ----------------------------------------------------------------------------
print("\n--- 2.1 Understanding self ---")

class Person:
    """Person class demonstrating self."""
    
    def __init__(self, name, age):
        """Initialize person."""
        # self refers to the instance being created
        self.name = name  # Instance variable
        self.age = age    # Instance variable
    
    def introduce(self):
        """Instance method using self."""
        # self is used to access instance variables
        return f"My name is {self.name} and I'm {self.age} years old"

person = Person("Alice", 25)
print(f"Introduction: {person.introduce()}")
print(f"Name: {person.name}")
print(f"Age: {person.age}")


# ----------------------------------------------------------------------------
# 2.2 Instance Attributes
# ----------------------------------------------------------------------------
print("\n--- 2.2 Instance Attributes ---")

class Person:
    """Person class with instance attributes."""
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.city = "Unknown"  # Can set default values
    
    def set_city(self, city):
        """Set city attribute."""
        self.city = city

person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

# Each instance has its own attributes
print(f"Person 1: {person1.name}, {person1.age}, {person1.city}")
print(f"Person 2: {person2.name}, {person2.age}, {person2.city}")

# Modify instance attributes
person1.set_city("New York")
person2.city = "London"  # Can also set directly

print(f"\nAfter modification:")
print(f"Person 1: {person1.name}, {person1.city}")
print(f"Person 2: {person2.name}, {person2.city}")


# ----------------------------------------------------------------------------
# 2.3 Adding Attributes Dynamically
# ----------------------------------------------------------------------------
print("\n--- 2.3 Adding Attributes Dynamically ---")

class Person:
    """Person class - attributes can be added dynamically."""
    
    def __init__(self, name):
        self.name = name

person = Person("Alice")
print(f"Initial: {person.name}")

# Add new attribute dynamically
person.email = "alice@example.com"
person.age = 25

print(f"After adding attributes:")
print(f"Name: {person.name}")
print(f"Email: {person.email}")
print(f"Age: {person.age}")


# ============================================================================
# 3. CLASS VARIABLES
# ============================================================================

print("\n" + "=" * 70)
print("3. CLASS VARIABLES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Class Variables vs Instance Variables
# ----------------------------------------------------------------------------
print("\n--- 3.1 Class Variables vs Instance Variables ---")

class Person:
    """Person class demonstrating class vs instance variables."""
    
    # Class variable - shared by all instances
    species = "Homo sapiens"
    population = 0
    
    def __init__(self, name, age):
        # Instance variables - unique to each instance
        self.name = name
        self.age = age
        Person.population += 1  # Access class variable via class name

person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

print(f"Person 1: {person1.name}, Species: {person1.species}")
print(f"Person 2: {person2.name}, Species: {person2.species}")
print(f"Population: {Person.population}")

# Class variable is shared
print(f"\nperson1.species == person2.species: {person1.species == person2.species}")
print(f"Person.species: {Person.species}")


# ----------------------------------------------------------------------------
# 3.2 Modifying Class Variables
# ----------------------------------------------------------------------------
print("\n--- 3.2 Modifying Class Variables ---")

class Person:
    """Person class - modifying class variables."""
    
    species = "Homo sapiens"
    
    def __init__(self, name):
        self.name = name

person1 = Person("Alice")
person2 = Person("Bob")

print(f"Initial species: {person1.species}")

# Modify via class
Person.species = "Human"
print(f"After Person.species = 'Human': {person1.species}")

# Create instance variable with same name (shadows class variable)
person1.species = "Alien"
print(f"\nAfter person1.species = 'Alien':")
print(f"person1.species: {person1.species}")
print(f"person2.species: {person2.species}")
print(f"Person.species: {Person.species}")


# ----------------------------------------------------------------------------
# 3.3 Class Variables - Common Patterns
# ----------------------------------------------------------------------------
print("\n--- 3.3 Class Variables - Common Patterns ---")

class Circle:
    """Circle class with class variable."""
    
    pi = 3.14159  # Class variable
    
    def __init__(self, radius):
        self.radius = radius  # Instance variable
    
    def area(self):
        """Calculate area using class variable."""
        return Circle.pi * self.radius ** 2
    
    @classmethod
    def set_pi(cls, value):
        """Class method to modify class variable."""
        cls.pi = value

circle1 = Circle(5)
circle2 = Circle(10)

print(f"Circle 1 area: {circle1.area()}")
print(f"Circle 2 area: {circle2.area()}")
print(f"Pi value: {Circle.pi}")

# Modify via class method
Circle.set_pi(3.14)
print(f"\nAfter setting pi to 3.14:")
print(f"Circle 1 area: {circle1.area()}")


# ============================================================================
# 4. METHODS
# ============================================================================

print("\n" + "=" * 70)
print("4. METHODS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Instance Methods
# ----------------------------------------------------------------------------
print("\n--- 4.1 Instance Methods ---")

class Person:
    """Person class with instance methods."""
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        """Instance method - takes self as first parameter."""
        return f"Hi, I'm {self.name} and I'm {self.age} years old"
    
    def have_birthday(self):
        """Instance method that modifies instance state."""
        self.age += 1
        return f"Happy birthday! Now I'm {self.age} years old"

person = Person("Alice", 25)
print(f"Introduction: {person.introduce()}")
print(f"Birthday: {person.have_birthday()}")
print(f"After birthday: {person.introduce()}")


# ----------------------------------------------------------------------------
# 4.2 Class Methods (@classmethod)
# ----------------------------------------------------------------------------
print("\n--- 4.2 Class Methods (@classmethod) ---")

class Person:
    """Person class with class methods."""
    
    population = 0
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
        Person.population += 1
    
    @classmethod
    def get_population(cls):
        """Class method - receives class as first parameter."""
        return cls.population
    
    @classmethod
    def from_birth_year(cls, name, birth_year):
        """Alternative constructor using class method."""
        import datetime
        current_year = datetime.datetime.now().year
        age = current_year - birth_year
        return cls(name, age)  # Create new instance
    
    @classmethod
    def create_baby(cls, name):
        """Class method to create baby."""
        return cls(name, 0)

person1 = Person("Alice", 25)
person2 = Person.from_birth_year("Bob", 1990)
person3 = Person.create_baby("Charlie")

print(f"Person 1: {person1.name}, {person1.age}")
print(f"Person 2: {person2.name}, {person2.age}")
print(f"Person 3: {person3.name}, {person3.age}")
print(f"Population: {Person.get_population()}")


# ----------------------------------------------------------------------------
# 4.3 Static Methods (@staticmethod)
# ----------------------------------------------------------------------------
print("\n--- 4.3 Static Methods (@staticmethod) ---")

class Person:
    """Person class with static methods."""
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    @staticmethod
    def is_adult(age):
        """Static method - no self or cls parameter."""
        return age >= 18
    
    @staticmethod
    def calculate_age_difference(age1, age2):
        """Static method - utility function."""
        return abs(age1 - age2)
    
    @staticmethod
    def validate_age(age):
        """Static method for validation."""
        return 0 <= age <= 150

person = Person("Alice", 25)

# Call static methods via class
print(f"Is 25 an adult? {Person.is_adult(25)}")
print(f"Is 15 an adult? {Person.is_adult(15)}")
print(f"Age difference: {Person.calculate_age_difference(25, 30)}")
print(f"Valid age? {Person.validate_age(25)}")

# Can also call via instance
print(f"\nVia instance: {person.is_adult(25)}")


# ----------------------------------------------------------------------------
# 4.4 Comparing Method Types
# ----------------------------------------------------------------------------
print("\n--- 4.4 Comparing Method Types ---")
print("""
INSTANCE METHODS:
- First parameter: self (instance)
- Access: instance.method()
- Can access: instance variables, class variables
- Use: Operations on instance data

CLASS METHODS:
- First parameter: cls (class)
- Access: Class.method() or instance.method()
- Can access: Class variables, create instances
- Use: Alternative constructors, class-level operations

STATIC METHODS:
- No special first parameter
- Access: Class.method() or instance.method()
- Cannot access: Instance or class variables directly
- Use: Utility functions related to class
""")


# ============================================================================
# 5. MAGIC METHODS
# ============================================================================

print("\n" + "=" * 70)
print("5. MAGIC METHODS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 __str__ and __repr__
# ----------------------------------------------------------------------------
print("\n--- 5.1 __str__ and __repr__ ---")

class Person:
    """Person class with __str__ and __repr__."""
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __str__(self):
        """String representation for end users (readable)."""
        return f"{self.name}, {self.age} years old"
    
    def __repr__(self):
        """String representation for developers (unambiguous)."""
        return f"Person(name='{self.name}', age={self.age})"

person = Person("Alice", 25)

# __str__ is used by print() and str()
print(f"print(): {person}")
print(f"str(): {str(person)}")

# __repr__ is used by repr() and in containers
print(f"\nrepr(): {repr(person)}")
print(f"In list: {[person]}")

# If __str__ not defined, __repr__ is used
class SimplePerson:
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f"SimplePerson('{self.name}')"

simple = SimplePerson("Bob")
print(f"\nSimplePerson: {simple}")


# ----------------------------------------------------------------------------
# 5.2 __len__
# ----------------------------------------------------------------------------
print("\n--- 5.2 __len__ ---")

class ShoppingCart:
    """Shopping cart with __len__ method."""
    
    def __init__(self):
        self.items = []
    
    def add_item(self, item):
        """Add item to cart."""
        self.items.append(item)
    
    def __len__(self):
        """Return number of items."""
        return len(self.items)

cart = ShoppingCart()
cart.add_item("Apple")
cart.add_item("Banana")
cart.add_item("Cherry")

print(f"Cart items: {cart.items}")
print(f"Length: {len(cart)}")


# ----------------------------------------------------------------------------
# 5.3 __eq__ (Equality)
# ----------------------------------------------------------------------------
print("\n--- 5.3 __eq__ (Equality) ---")

class Point:
    """Point class with __eq__ method."""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        """Check equality."""
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

p1 = Point(1, 2)
p2 = Point(1, 2)
p3 = Point(3, 4)

print(f"p1: {p1}")
print(f"p2: {p2}")
print(f"p3: {p3}")
print(f"\np1 == p2: {p1 == p2}")
print(f"p1 == p3: {p1 == p3}")
print(f"p1 != p3: {p1 != p3}")  # Uses __eq__ automatically


# ----------------------------------------------------------------------------
# 5.4 __lt__ (Less Than)
# ----------------------------------------------------------------------------
print("\n--- 5.4 __lt__ (Less Than) ---")

class Person:
    """Person class with __lt__ method."""
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __lt__(self, other):
        """Compare by age."""
        if not isinstance(other, Person):
            return NotImplemented
        return self.age < other.age
    
    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

person1 = Person("Alice", 25)
person2 = Person("Bob", 30)
person3 = Person("Charlie", 20)

print(f"person1: {person1}")
print(f"person2: {person2}")
print(f"person3: {person3}")
print(f"\nperson1 < person2: {person1 < person2}")
print(f"person3 < person1: {person3 < person1}")

# Can sort objects
people = [person1, person2, person3]
sorted_people = sorted(people)
print(f"\nSorted by age: {sorted_people}")


# ----------------------------------------------------------------------------
# 5.5 __hash__
# ----------------------------------------------------------------------------
print("\n--- 5.5 __hash__ ---")

class Point:
    """Point class with __hash__ method."""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        """Equality check."""
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        """Hash function - must be consistent with __eq__."""
        return hash((self.x, self.y))
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

p1 = Point(1, 2)
p2 = Point(1, 2)
p3 = Point(3, 4)

print(f"p1: {p1}, hash: {hash(p1)}")
print(f"p2: {p2}, hash: {hash(p2)}")
print(f"p3: {p3}, hash: {hash(p3)}")
print(f"\np1 == p2: {p1 == p2}")
print(f"hash(p1) == hash(p2): {hash(p1) == hash(p2)}")

# Can use in sets and as dictionary keys
points_set = {p1, p2, p3}
print(f"\nPoints set: {points_set}")  # p1 and p2 are equal, so only one in set

points_dict = {p1: "first", p3: "third"}
print(f"Points dict: {points_dict}")


# ----------------------------------------------------------------------------
# 5.6 Complete Example with Multiple Magic Methods
# ----------------------------------------------------------------------------
print("\n--- 5.6 Complete Example with Multiple Magic Methods ---")

class Student:
    """Student class with multiple magic methods."""
    
    def __init__(self, name, student_id, gpa):
        self.name = name
        self.student_id = student_id
        self.gpa = gpa
    
    def __str__(self):
        """User-friendly string."""
        return f"{self.name} (ID: {self.student_id}, GPA: {self.gpa})"
    
    def __repr__(self):
        """Developer-friendly string."""
        return f"Student(name='{self.name}', student_id={self.student_id}, gpa={self.gpa})"
    
    def __eq__(self, other):
        """Equality by student ID."""
        if not isinstance(other, Student):
            return NotImplemented
        return self.student_id == other.student_id
    
    def __lt__(self, other):
        """Compare by GPA (higher GPA is 'less' for sorting ascending)."""
        if not isinstance(other, Student):
            return NotImplemented
        return self.gpa < other.gpa
    
    def __hash__(self):
        """Hash by student ID."""
        return hash(self.student_id)
    
    def __len__(self):
        """Return length of name."""
        return len(self.name)

student1 = Student("Alice", 1001, 3.8)
student2 = Student("Bob", 1002, 3.5)
student3 = Student("Charlie", 1001, 3.9)  # Same ID as student1

print(f"Student 1: {student1}")
print(f"Student 2: {student2}")
print(f"Student 3: {student3}")
print(f"\nstudent1 == student3 (same ID): {student1 == student3}")
print(f"student1 < student2 (by GPA): {student1 < student2}")
print(f"len(student1): {len(student1)}")

# Can use in sets (hashable)
students_set = {student1, student2, student3}
print(f"\nStudents set (unique by ID): {students_set}")

# Can sort
students_list = [student1, student2, student3]
sorted_students = sorted(students_list)
print(f"\nSorted by GPA: {sorted_students}")


# ============================================================================
# 6. PRACTICAL EXAMPLES
# ============================================================================

print("\n" + "=" * 70)
print("6. PRACTICAL EXAMPLES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 6.1 Bank Account Example
# ----------------------------------------------------------------------------
print("\n--- 6.1 Bank Account Example ---")

class BankAccount:
    """Bank account with instance and class variables."""
    
    # Class variable
    interest_rate = 0.05
    account_count = 0
    
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
        BankAccount.account_count += 1
        self.account_number = BankAccount.account_count
    
    def deposit(self, amount):
        """Deposit money."""
        self.balance += amount
        return self.balance
    
    def withdraw(self, amount):
        """Withdraw money."""
        if amount <= self.balance:
            self.balance -= amount
            return self.balance
        return "Insufficient funds"
    
    @classmethod
    def set_interest_rate(cls, rate):
        """Set interest rate for all accounts."""
        cls.interest_rate = rate
    
    @staticmethod
    def calculate_interest(principal, rate, years):
        """Calculate compound interest."""
        return principal * ((1 + rate) ** years - 1)
    
    def __str__(self):
        return f"Account {self.account_number}: {self.owner}, Balance: ${self.balance:.2f}"
    
    def __repr__(self):
        return f"BankAccount(owner='{self.owner}', balance={self.balance})"

account1 = BankAccount("Alice", 1000)
account2 = BankAccount("Bob", 500)

print(f"Account 1: {account1}")
print(f"Account 2: {account2}")
print(f"Total accounts: {BankAccount.account_count}")

account1.deposit(200)
print(f"\nAfter deposit: {account1}")

interest = BankAccount.calculate_interest(1000, 0.05, 5)
print(f"Interest on $1000: ${interest:.2f}")


# ============================================================================
# 7. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("7. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. CLASS DEFINITION:
   - Use 'class' keyword
   - __init__ is constructor (not required but common)
   - self is first parameter in instance methods
   - Can have default parameters

2. INSTANCE VARIABLES:
   - Defined with self.variable_name
   - Unique to each instance
   - Can be added/modified after creation
   - Accessed via instance.variable_name

3. CLASS VARIABLES:
   - Defined at class level (outside __init__)
   - Shared by all instances
   - Access via Class.variable or instance.variable
   - Modifying via instance creates instance variable (shadows class variable)

4. METHODS:
   - Instance: def method(self, ...) - operates on instance
   - Class: @classmethod def method(cls, ...) - operates on class
   - Static: @staticmethod def method(...) - utility function
   - self/cls are conventions, not keywords

5. MAGIC METHODS:
   - __str__: User-friendly string (print, str())
   - __repr__: Developer string (repr(), containers)
   - __len__: len() function
   - __eq__: == operator
   - __lt__: < operator (enables sorting)
   - __hash__: hash() function (needed for sets/dict keys)
   - If __eq__ defined, __hash__ must be defined (or set __hash__ = None)

6. BEST PRACTICES:
   - Always define __repr__ (at minimum)
   - __str__ should be readable
   - __repr__ should be unambiguous
   - __eq__ and __hash__ must be consistent
   - Use class methods for alternative constructors
   - Use static methods for utility functions

7. COMMON PATTERNS:
   - Class variables for shared state
   - Instance variables for unique state
   - Class methods for factory methods
   - Static methods for utilities
   - Magic methods for operator overloading

8. INTERVIEW TIPS:
   - Know difference between class and instance variables
   - Understand when to use @classmethod vs @staticmethod
   - Know which magic methods are commonly used
   - Be able to implement __eq__, __hash__, __str__, __repr__
   - Understand self and cls parameters
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Classes & Objects Guide Ready!")
    print("=" * 70)
