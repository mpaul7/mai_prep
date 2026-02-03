"""
Python Inheritance - Interview Preparation
Topic 3.2: Inheritance

This module covers:
- Single Inheritance: Base classes, derived classes
- Multiple Inheritance: Method Resolution Order (MRO)
- Method Overriding: super() function
- Abstract Base Classes: abc module basics
"""

from abc import ABC, abstractmethod

# ============================================================================
# 1. SINGLE INHERITANCE
# ============================================================================

print("=" * 70)
print("1. SINGLE INHERITANCE")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Basic Inheritance
# ----------------------------------------------------------------------------
print("\n--- 1.1 Basic Inheritance ---")

class Animal:
    """Base class (parent class)."""
    
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def speak(self):
        return f"{self.name} makes a sound"
    
    def move(self):
        return f"{self.name} moves"

class Dog(Animal):
    """Derived class (child class) inheriting from Animal."""
    
    def __init__(self, name, breed):
        # Call parent's __init__
        super().__init__(name, "Dog")
        self.breed = breed
    
    def speak(self):
        """Override parent's speak method."""
        return f"{self.name} barks"
    
    def fetch(self):
        """New method specific to Dog."""
        return f"{self.name} fetches the ball"

# Create instances
animal = Animal("Generic", "Unknown")
dog = Dog("Buddy", "Golden Retriever")

print(f"Animal: {animal.speak()}")
print(f"Dog: {dog.speak()}")
print(f"Dog move: {dog.move()}")  # Inherited from Animal
print(f"Dog fetch: {dog.fetch()}")  # Dog-specific method


# ----------------------------------------------------------------------------
# 1.2 Accessing Parent Methods
# ----------------------------------------------------------------------------
print("\n--- 1.2 Accessing Parent Methods ---")

class Vehicle:
    """Base class for vehicles."""
    
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    
    def start(self):
        return f"{self.brand} {self.model} starts"
    
    def stop(self):
        return f"{self.brand} {self.model} stops"

class Car(Vehicle):
    """Car inheriting from Vehicle."""
    
    def __init__(self, brand, model, fuel_type):
        super().__init__(brand, model)
        self.fuel_type = fuel_type
    
    def start(self):
        # Call parent method and extend it
        parent_result = super().start()
        return f"{parent_result} with {self.fuel_type} engine"

car = Car("Toyota", "Camry", "gasoline")
print(f"Car start: {car.start()}")
print(f"Car stop: {car.stop()}")  # Inherited method


# ----------------------------------------------------------------------------
# 1.3 Inheriting Attributes
# ----------------------------------------------------------------------------
print("\n--- 1.3 Inheriting Attributes ---")

class Person:
    """Base class."""
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"I'm {self.name}, {self.age} years old"

class Student(Person):
    """Student inheriting from Person."""
    
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id
    
    def introduce(self):
        # Use parent's introduce and extend it
        parent_intro = super().introduce()
        return f"{parent_intro} (Student ID: {self.student_id})"

student = Student("Alice", 20, "S12345")
print(f"Student intro: {student.introduce()}")
print(f"Student name: {student.name}")  # Inherited attribute
print(f"Student ID: {student.student_id}")  # Student-specific attribute


# ============================================================================
# 2. METHOD OVERRIDING
# ============================================================================

print("\n" + "=" * 70)
print("2. METHOD OVERRIDING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Basic Method Overriding
# ----------------------------------------------------------------------------
print("\n--- 2.1 Basic Method Overriding ---")

class Shape:
    """Base class."""
    
    def area(self):
        """To be overridden by subclasses."""
        raise NotImplementedError("Subclass must implement area()")
    
    def perimeter(self):
        """To be overridden by subclasses."""
        raise NotImplementedError("Subclass must implement perimeter()")

class Rectangle(Shape):
    """Rectangle class overriding Shape methods."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        """Override parent's area method."""
        return self.width * self.height
    
    def perimeter(self):
        """Override parent's perimeter method."""
        return 2 * (self.width + self.height)

class Circle(Shape):
    """Circle class overriding Shape methods."""
    
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        """Override parent's area method."""
        import math
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        """Override parent's perimeter method."""
        import math
        return 2 * math.pi * self.radius

rect = Rectangle(5, 3)
circle = Circle(4)

print(f"Rectangle area: {rect.area()}")
print(f"Rectangle perimeter: {rect.perimeter()}")
print(f"Circle area: {circle.area():.2f}")
print(f"Circle perimeter: {circle.perimeter():.2f}")


# ----------------------------------------------------------------------------
# 2.2 Using super() for Method Overriding
# ----------------------------------------------------------------------------
print("\n--- 2.2 Using super() for Method Overriding ---")

class Employee:
    """Base class."""
    
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
    
    def get_info(self):
        return f"Employee: {self.name}, Salary: ${self.salary}"

class Manager(Employee):
    """Manager inheriting from Employee."""
    
    def __init__(self, name, salary, department):
        super().__init__(name, salary)
        self.department = department
    
    def get_info(self):
        # Call parent method and extend it
        parent_info = super().get_info()
        return f"{parent_info}, Department: {self.department}"

manager = Manager("John", 100000, "Engineering")
print(f"Manager info: {manager.get_info()}")


# ----------------------------------------------------------------------------
# 2.3 super() in Different Contexts
# ----------------------------------------------------------------------------
print("\n--- 2.3 super() in Different Contexts ---")

class Base:
    """Base class."""
    
    def __init__(self):
        print("Base __init__")
    
    def method(self):
        print("Base method")

class Derived(Base):
    """Derived class."""
    
    def __init__(self):
        super().__init__()  # Call parent __init__
        print("Derived __init__")
    
    def method(self):
        super().method()  # Call parent method
        print("Derived method")

derived = Derived()
print()
derived.method()


# ============================================================================
# 3. MULTIPLE INHERITANCE
# ============================================================================

print("\n" + "=" * 70)
print("3. MULTIPLE INHERITANCE")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Basic Multiple Inheritance
# ----------------------------------------------------------------------------
print("\n--- 3.1 Basic Multiple Inheritance ---")

class Flyable:
    """Mixin class for flying capability."""
    
    def fly(self):
        return "Flying"

class Swimmable:
    """Mixin class for swimming capability."""
    
    def swim(self):
        return "Swimming"

class Duck(Flyable, Swimmable):
    """Duck inherits from both Flyable and Swimmable."""
    
    def __init__(self, name):
        self.name = name
    
    def quack(self):
        return f"{self.name} quacks"

duck = Duck("Donald")
print(f"Duck fly: {duck.fly()}")
print(f"Duck swim: {duck.swim()}")
print(f"Duck quack: {duck.quack()}")


# ----------------------------------------------------------------------------
# 3.2 Method Resolution Order (MRO)
# ----------------------------------------------------------------------------
print("\n--- 3.2 Method Resolution Order (MRO) ---")

class A:
    """Class A."""
    
    def method(self):
        return "A.method()"

class B(A):
    """Class B inheriting from A."""
    
    def method(self):
        return "B.method()"

class C(A):
    """Class C inheriting from A."""
    
    def method(self):
        return "C.method()"

class D(B, C):
    """Class D inheriting from both B and C."""
    pass

# Check MRO
print(f"D.__mro__: {D.__mro__}")
print(f"D.mro(): {D.mro()}")

d = D()
print(f"\nd.method(): {d.method()}")  # Calls B.method() due to MRO

# MRO follows: D -> B -> C -> A -> object
# So B.method() is called first


# ----------------------------------------------------------------------------
# 3.3 Diamond Problem
# ----------------------------------------------------------------------------
print("\n--- 3.3 Diamond Problem ---")

class Grandparent:
    """Grandparent class."""
    
    def method(self):
        return "Grandparent.method()"

class Parent1(Grandparent):
    """Parent 1."""
    
    def method(self):
        result = super().method()
        return f"Parent1.method() -> {result}"

class Parent2(Grandparent):
    """Parent 2."""
    
    def method(self):
        result = super().method()
        return f"Parent2.method() -> {result}"

class Child(Parent1, Parent2):
    """Child inheriting from both parents."""
    
    def method(self):
        result = super().method()
        return f"Child.method() -> {result}"

child = Child()
print(f"Child MRO: {Child.__mro__}")
print(f"\nchild.method(): {child.method()}")

# MRO ensures each class is only called once
# Order: Child -> Parent1 -> Parent2 -> Grandparent -> object


# ----------------------------------------------------------------------------
# 3.4 Using super() with Multiple Inheritance
# ----------------------------------------------------------------------------
print("\n--- 3.4 Using super() with Multiple Inheritance ---")

class A:
    def __init__(self):
        print("A.__init__")
        super().__init__()

class B:
    def __init__(self):
        print("B.__init__")
        super().__init__()

class C(A, B):
    def __init__(self):
        print("C.__init__")
        super().__init__()

c = C()
print(f"\nC MRO: {C.__mro__}")

# super() follows MRO, so:
# C.__init__ -> A.__init__ -> B.__init__ -> object.__init__


# ============================================================================
# 4. ABSTRACT BASE CLASSES
# ============================================================================

print("\n" + "=" * 70)
print("4. ABSTRACT BASE CLASSES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Basic Abstract Base Class
# ----------------------------------------------------------------------------
print("\n--- 4.1 Basic Abstract Base Class ---")

class Shape(ABC):
    """Abstract base class for shapes."""
    
    @abstractmethod
    def area(self):
        """Abstract method - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Abstract method - must be implemented by subclasses."""
        pass
    
    def describe(self):
        """Concrete method - can be used by subclasses."""
        return f"Shape with area {self.area():.2f}"

class Rectangle(Shape):
    """Concrete implementation of Shape."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        """Implement abstract method."""
        return self.width * self.height
    
    def perimeter(self):
        """Implement abstract method."""
        return 2 * (self.width + self.height)

class Circle(Shape):
    """Concrete implementation of Shape."""
    
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        """Implement abstract method."""
        import math
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        """Implement abstract method."""
        import math
        return 2 * math.pi * self.radius

# Cannot instantiate abstract class
# shape = Shape()  # Would raise TypeError

# Can instantiate concrete implementations
rect = Rectangle(5, 3)
circle = Circle(4)

print(f"Rectangle: {rect.describe()}")
print(f"Circle: {circle.describe()}")


# ----------------------------------------------------------------------------
# 4.2 Abstract Methods with Implementation
# ----------------------------------------------------------------------------
print("\n--- 4.2 Abstract Methods with Implementation ---")

class Animal(ABC):
    """Abstract base class for animals."""
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def speak(self):
        """Abstract method - must be implemented."""
        pass
    
    def introduce(self):
        """Concrete method using abstract method."""
        return f"I'm {self.name} and I say: {self.speak()}"

class Dog(Animal):
    """Concrete implementation."""
    
    def speak(self):
        return "Woof!"

class Cat(Animal):
    """Concrete implementation."""
    
    def speak(self):
        return "Meow!"

dog = Dog("Buddy")
cat = Cat("Whiskers")

print(f"Dog: {dog.introduce()}")
print(f"Cat: {cat.introduce()}")


# ----------------------------------------------------------------------------
# 4.3 Abstract Properties
# ----------------------------------------------------------------------------
print("\n--- 4.3 Abstract Properties ---")

class Vehicle(ABC):
    """Abstract base class for vehicles."""
    
    @property
    @abstractmethod
    def max_speed(self):
        """Abstract property - must be implemented."""
        pass
    
    @abstractmethod
    def start(self):
        """Abstract method."""
        pass

class Car(Vehicle):
    """Concrete implementation."""
    
    def __init__(self, brand, max_speed_value):
        self.brand = brand
        self._max_speed = max_speed_value
    
    @property
    def max_speed(self):
        """Implement abstract property."""
        return self._max_speed
    
    def start(self):
        """Implement abstract method."""
        return f"{self.brand} starts"

car = Car("Toyota", 120)
print(f"Car: {car.start()}")
print(f"Max speed: {car.max_speed} mph")


# ----------------------------------------------------------------------------
# 4.4 Mixing Abstract and Concrete Methods
# ----------------------------------------------------------------------------
print("\n--- 4.4 Mixing Abstract and Concrete Methods ---")

class Database(ABC):
    """Abstract database interface."""
    
    @abstractmethod
    def connect(self):
        """Abstract method - must be implemented."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Abstract method - must be implemented."""
        pass
    
    def execute_query(self, query):
        """Concrete method using abstract methods."""
        self.connect()
        result = f"Executing: {query}"
        self.disconnect()
        return result

class MySQLDatabase(Database):
    """MySQL implementation."""
    
    def connect(self):
        return "Connected to MySQL"
    
    def disconnect(self):
        return "Disconnected from MySQL"

class PostgreSQLDatabase(Database):
    """PostgreSQL implementation."""
    
    def connect(self):
        return "Connected to PostgreSQL"
    
    def disconnect(self):
        return "Disconnected from PostgreSQL"

mysql = MySQLDatabase()
postgres = PostgreSQLDatabase()

print(f"MySQL: {mysql.execute_query('SELECT * FROM users')}")
print(f"PostgreSQL: {postgres.execute_query('SELECT * FROM users')}")


# ============================================================================
# 5. PRACTICAL EXAMPLES
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICAL EXAMPLES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 Employee Hierarchy
# ----------------------------------------------------------------------------
print("\n--- 5.1 Employee Hierarchy ---")

class Employee:
    """Base employee class."""
    
    def __init__(self, name, employee_id):
        self.name = name
        self.employee_id = employee_id
    
    def get_info(self):
        return f"Employee {self.employee_id}: {self.name}"

class Manager(Employee):
    """Manager inheriting from Employee."""
    
    def __init__(self, name, employee_id, department):
        super().__init__(name, employee_id)
        self.department = department
    
    def get_info(self):
        base_info = super().get_info()
        return f"{base_info}, Manager of {self.department}"

class Developer(Employee):
    """Developer inheriting from Employee."""
    
    def __init__(self, name, employee_id, programming_language):
        super().__init__(name, employee_id)
        self.programming_language = programming_language
    
    def get_info(self):
        base_info = super().get_info()
        return f"{base_info}, Developer ({self.programming_language})"

manager = Manager("Alice", "M001", "Engineering")
developer = Developer("Bob", "D001", "Python")

print(f"Manager: {manager.get_info()}")
print(f"Developer: {developer.get_info()}")


# ----------------------------------------------------------------------------
# 5.2 Shape Hierarchy with Abstract Base Class
# ----------------------------------------------------------------------------
print("\n--- 5.2 Shape Hierarchy with Abstract Base Class ---")

class Shape(ABC):
    """Abstract base class for shapes."""
    
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass
    
    def __str__(self):
        return f"{self.__class__.__name__} with area {self.area():.2f}"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Square(Rectangle):
    """Square inheriting from Rectangle."""
    
    def __init__(self, side):
        super().__init__(side, side)  # Square is rectangle with equal sides

rect = Rectangle(5, 3)
square = Square(4)

print(f"Rectangle: {rect}")
print(f"Square: {square}")


# ============================================================================
# 6. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("6. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. SINGLE INHERITANCE:
   - One parent class, one child class
   - Child inherits all attributes and methods
   - Use super() to call parent methods
   - Syntax: class Child(Parent):

2. METHOD OVERRIDING:
   - Redefine method in child class
   - Child method replaces parent method
   - Use super() to call parent method from child
   - Allows customization while reusing code

3. MULTIPLE INHERITANCE:
   - Class can inherit from multiple parents
   - Syntax: class Child(Parent1, Parent2):
   - Method Resolution Order (MRO) determines method lookup
   - MRO follows C3 linearization algorithm

4. METHOD RESOLUTION ORDER (MRO):
   - Determines order of method lookup
   - Check with Class.__mro__ or Class.mro()
   - Follows: Child -> Parent1 -> Parent2 -> ... -> object
   - Ensures each class in hierarchy called once
   - super() follows MRO

5. super() FUNCTION:
   - Calls method from parent class
   - Works with single and multiple inheritance
   - Follows MRO in multiple inheritance
   - Syntax: super().method() or super(Class, self).method()

6. ABSTRACT BASE CLASSES:
   - Use ABC class and @abstractmethod decorator
   - Cannot instantiate abstract class
   - Forces subclasses to implement abstract methods
   - Use for defining interfaces/contracts
   - Import: from abc import ABC, abstractmethod

7. WHEN TO USE INHERITANCE:
   - "Is-a" relationship (Dog IS-A Animal)
   - Code reuse and polymorphism
   - When subclasses share common behavior
   - When need to enforce interface (ABC)

8. BEST PRACTICES:
   - Use single inheritance when possible
   - Be careful with multiple inheritance
   - Always call super().__init__() in child __init__
   - Use ABC for interfaces
   - Understand MRO for multiple inheritance
   - Prefer composition over inheritance when appropriate

9. INTERVIEW TIPS:
   - Know difference between single and multiple inheritance
   - Understand how MRO works
   - Know when to use super()
   - Understand abstract base classes
   - Be able to explain diamond problem
   - Know when inheritance vs composition
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Inheritance Guide Ready!")
    print("=" * 70)
