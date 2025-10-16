"""
Object-Oriented Programming in Python for Data Science Interview Preparation

This module covers comprehensive OOP concepts commonly tested in data science interviews,
particularly focusing on magic methods, inheritance, and polymorphism.

Topics covered:
1. Classes and Objects
2. Magic Methods (__init__, __str__, __repr__, __iter__, __len__, etc.)
3. Inheritance (Single, Multiple, Method Resolution Order)
4. Polymorphism (Method Overriding, Duck Typing)
5. Encapsulation (Private/Protected attributes)
6. Abstract Base Classes
7. Property Decorators
8. Class Methods and Static Methods
9. Composition vs Inheritance
10. Design Patterns
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Iterator
import math
import copy
from functools import wraps


# ========== BASIC CLASSES AND MAGIC METHODS ==========

class Point:
    """
    A simple Point class demonstrating basic magic methods.
    """
    
    def __init__(self, x: float = 0, y: float = 0):
        """Initialize a point with x and y coordinates."""
        self.x = x
        self.y = y
    
    def _# It looks like the code snippet you provided is attempting to define a `__str__` method in a
    # Python class. However, there are a couple of issues in the code:
    _str__(self) -> str:
        """String representation for end users."""
        return f"Point({self.x}, {self.y})"
    
    def __repr__(self) -> str:
        """String representation for developers (should be unambiguous)."""
        return f"Point(x={self.x}, y={self.y})"
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y
    
    def __lt__(self, other) -> bool:
        """Less than comparison (by distance from origin)."""
        if not isinstance(other, Point):
            return NotImplemented
        return self.distance_from_origin() < other.distance_from_origin()
    
    def __add__(self, other):
        """Addition of two points."""
        if not isinstance(other, Point):
            return NotImplemented
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Subtraction of two points."""
        if not isinstance(other, Point):
            return NotImplemented
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: Union[int, float]):
        """Scalar multiplication."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Point(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar: Union[int, float]):
        """Right scalar multiplication."""
        return self.__mul__(scalar)
    
    def __abs__(self) -> float:
        """Absolute value (distance from origin)."""
        return self.distance_from_origin()
    
    def __hash__(self) -> int:
        """Hash function to make Point hashable."""
        return hash((self.x, self.y))
    
    def distance_from_origin(self) -> float:
        """Calculate distance from origin."""
        return math.sqrt(self.x**2 + self.y**2)
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class Vector:
    """
    A Vector class demonstrating more advanced magic methods.
    """
    
    def __init__(self, *components):
        """Initialize vector with components."""
        self.components = list(components)
    
    def __str__(self) -> str:
        """String representation."""
        return f"Vector{tuple(self.components)}"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"Vector({', '.join(map(str, self.components))})"
    
    def __len__(self) -> int:
        """Length of vector (number of components)."""
        return len(self.components)
    
    def __getitem__(self, index: int) -> float:
        """Get component by index."""
        return self.components[index]
    
    def __setitem__(self, index: int, value: float):
        """Set component by index."""
        self.components[index] = value
    
    def __iter__(self) -> Iterator[float]:
        """Make vector iterable."""
        return iter(self.components)
    
    def __contains__(self, value: float) -> bool:
        """Check if value is in vector."""
        return value in self.components
    
    def __add__(self, other):
        """Vector addition."""
        if not isinstance(other, Vector):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("Vectors must have same dimension")
        
        return Vector(*(a + b for a, b in zip(self, other)))
    
    def __sub__(self, other):
        """Vector subtraction."""
        if not isinstance(other, Vector):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("Vectors must have same dimension")
        
        return Vector(*(a - b for a, b in zip(self, other)))
    
    def __mul__(self, other):
        """Scalar multiplication or dot product."""
        if isinstance(other, (int, float)):
            # Scalar multiplication
            return Vector(*(component * other for component in self))
        elif isinstance(other, Vector):
            # Dot product
            if len(self) != len(other):
                raise ValueError("Vectors must have same dimension")
            return sum(a * b for a, b in zip(self, other))
        return NotImplemented
    
    def __rmul__(self, scalar):
        """Right scalar multiplication."""
        return self.__mul__(scalar)
    
    def __abs__(self) -> float:
        """Magnitude of vector."""
        return math.sqrt(sum(component**2 for component in self))
    
    def __bool__(self) -> bool:
        """Truth value (False if zero vector)."""
        return any(component != 0 for component in self)
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, Vector):
            return NotImplemented
        return self.components == other.components


class Matrix:
    """
    A Matrix class demonstrating container magic methods.
    """
    
    def __init__(self, data: List[List[float]]):
        """Initialize matrix with 2D list."""
        if not data or not data[0]:
            raise ValueError("Matrix cannot be empty")
        
        rows = len(data)
        cols = len(data[0])
        
        # Validate that all rows have same length
        for row in data:
            if len(row) != cols:
                raise ValueError("All rows must have same length")
        
        self.data = [row[:] for row in data]  # Deep copy
        self.rows = rows
        self.cols = cols
    
    def __str__(self) -> str:
        """String representation."""
        return '\n'.join([' '.join(f'{val:6.2f}' for val in row) for row in self.data])
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"Matrix({self.data})"
    
    def __getitem__(self, key):
        """Get item by index (supports matrix[i][j] and matrix[i, j])."""
        if isinstance(key, tuple):
            row, col = key
            return self.data[row][col]
        else:
            return self.data[key]
    
    def __setitem__(self, key, value):
        """Set item by index."""
        if isinstance(key, tuple):
            row, col = key
            self.data[row][col] = value
        else:
            self.data[key] = value
    
    def __iter__(self):
        """Iterate over rows."""
        return iter(self.data)
    
    def __len__(self) -> int:
        """Number of rows."""
        return self.rows
    
    def __add__(self, other):
        """Matrix addition."""
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have same dimensions")
        
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j] + other.data[i][j])
            result.append(row)
        
        return Matrix(result)
    
    def __mul__(self, other):
        """Matrix multiplication or scalar multiplication."""
        if isinstance(other, (int, float)):
            # Scalar multiplication
            result = []
            for row in self.data:
                result.append([val * other for val in row])
            return Matrix(result)
        
        elif isinstance(other, Matrix):
            # Matrix multiplication
            if self.cols != other.rows:
                raise ValueError("Number of columns in first matrix must equal number of rows in second")
            
            result = []
            for i in range(self.rows):
                row = []
                for j in range(other.cols):
                    sum_val = 0
                    for k in range(self.cols):
                        sum_val += self.data[i][k] * other.data[k][j]
                    row.append(sum_val)
                result.append(row)
            
            return Matrix(result)
        
        return NotImplemented
    
    def __rmul__(self, scalar):
        """Right scalar multiplication."""
        return self.__mul__(scalar)
    
    def transpose(self):
        """Return transpose of matrix."""
        result = []
        for j in range(self.cols):
            row = []
            for i in range(self.rows):
                row.append(self.data[i][j])
            result.append(row)
        return Matrix(result)


# ========== INHERITANCE EXAMPLES ==========

class Animal:
    """Base class for animal hierarchy."""
    
    def __init__(self, name: str, species: str):
        """Initialize animal with name and species."""
        self.name = name
        self.species = species
        self._energy = 100  # Protected attribute
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name} the {self.species}"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"{self.__class__.__name__}(name='{self.name}', species='{self.species}')"
    
    def speak(self) -> str:
        """Make a sound (to be overridden)."""
        return "Some generic animal sound"
    
    def move(self) -> str:
        """Move (to be overridden)."""
        return f"{self.name} moves"
    
    def eat(self, food: str) -> str:
        """Eat food."""
        self._energy += 10
        return f"{self.name} eats {food}"
    
    def sleep(self) -> str:
        """Sleep to restore energy."""
        self._energy = 100
        return f"{self.name} sleeps and restores energy"
    
    @property
    def energy(self) -> int:
        """Get current energy level."""
        return self._energy
    
    def is_tired(self) -> bool:
        """Check if animal is tired."""
        return self._energy < 30


class Mammal(Animal):
    """Mammal class inheriting from Animal."""
    
    def __init__(self, name: str, species: str, fur_color: str = "brown"):
        """Initialize mammal."""
        super().__init__(name, species)
        self.fur_color = fur_color
        self.warm_blooded = True
    
    def give_birth(self) -> str:
        """Give birth (mammal-specific behavior)."""
        return f"{self.name} gives birth to live young"
    
    def produce_milk(self) -> str:
        """Produce milk (mammal-specific behavior)."""
        return f"{self.name} produces milk"


class Bird(Animal):
    """Bird class inheriting from Animal."""
    
    def __init__(self, name: str, species: str, can_fly: bool = True):
        """Initialize bird."""
        super().__init__(name, species)
        self.can_fly = can_fly
        self.has_feathers = True
    
    def move(self) -> str:
        """Override move method."""
        if self.can_fly:
            return f"{self.name} flies through the air"
        else:
            return f"{self.name} walks on the ground"
    
    def lay_eggs(self) -> str:
        """Lay eggs (bird-specific behavior)."""
        return f"{self.name} lays eggs"
    
    def build_nest(self) -> str:
        """Build nest."""
        return f"{self.name} builds a nest"


class Dog(Mammal):
    """Dog class inheriting from Mammal."""
    
    def __init__(self, name: str, breed: str = "Mixed"):
        """Initialize dog."""
        super().__init__(name, "Canis lupus", "varies")
        self.breed = breed
        self.loyalty = 100
    
    def speak(self) -> str:
        """Override speak method."""
        return f"{self.name} barks: Woof! Woof!"
    
    def move(self) -> str:
        """Override move method."""
        return f"{self.name} runs on four legs"
    
    def fetch(self, item: str) -> str:
        """Fetch item (dog-specific behavior)."""
        self._energy -= 5
        return f"{self.name} fetches the {item}"
    
    def wag_tail(self) -> str:
        """Wag tail."""
        return f"{self.name} wags tail happily"


class Cat(Mammal):
    """Cat class inheriting from Mammal."""
    
    def __init__(self, name: str, breed: str = "Domestic"):
        """Initialize cat."""
        super().__init__(name, "Felis catus", "varies")
        self.breed = breed
        self.independence = 90
    
    def speak(self) -> str:
        """Override speak method."""
        return f"{self.name} meows: Meow!"
    
    def move(self) -> str:
        """Override move method."""
        return f"{self.name} prowls silently"
    
    def purr(self) -> str:
        """Purr (cat-specific behavior)."""
        return f"{self.name} purrs contentedly"
    
    def climb(self, object_name: str) -> str:
        """Climb object."""
        return f"{self.name} climbs the {object_name}"


class Eagle(Bird):
    """Eagle class inheriting from Bird."""
    
    def __init__(self, name: str):
        """Initialize eagle."""
        super().__init__(name, "Aquila chrysaetos", can_fly=True)
        self.sharp_talons = True
        self.keen_eyesight = True
    
    def speak(self) -> str:
        """Override speak method."""
        return f"{self.name} screeches: Screech!"
    
    def hunt(self, prey: str) -> str:
        """Hunt prey."""
        self._energy -= 15
        return f"{self.name} hunts {prey} with sharp talons"
    
    def soar(self) -> str:
        """Soar high."""
        return f"{self.name} soars high in the sky"


# ========== MULTIPLE INHERITANCE ==========

class Swimmer:
    """Mixin class for swimming ability."""
    
    def swim(self) -> str:
        """Swim in water."""
        return f"{self.name} swims in the water"
    
    def dive(self, depth: int) -> str:
        """Dive to specified depth."""
        return f"{self.name} dives to {depth} meters"


class Flyer:
    """Mixin class for flying ability."""
    
    def fly(self) -> str:
        """Fly in the air."""
        return f"{self.name} flies through the air"
    
    def land(self) -> str:
        """Land on ground."""
        return f"{self.name} lands gracefully"


class Duck(Bird, Swimmer):
    """Duck class with multiple inheritance."""
    
    def __init__(self, name: str):
        """Initialize duck."""
        super().__init__(name, "Anas platyrhynchos", can_fly=True)
        self.waterproof_feathers = True
    
    def speak(self) -> str:
        """Override speak method."""
        return f"{self.name} quacks: Quack! Quack!"
    
    def move(self) -> str:
        """Override move method."""
        return f"{self.name} waddles on land and swims in water"


class Penguin(Bird, Swimmer):
    """Penguin class with multiple inheritance."""
    
    def __init__(self, name: str):
        """Initialize penguin."""
        super().__init__(name, "Spheniscidae", can_fly=False)
        self.insulation = "excellent"
    
    def speak(self) -> str:
        """Override speak method."""
        return f"{self.name} makes penguin sounds"
    
    def move(self) -> str:
        """Override move method."""
        return f"{self.name} waddles on ice and swims underwater"
    
    def slide_on_belly(self) -> str:
        """Slide on belly (penguin-specific)."""
        return f"{self.name} slides on belly across the ice"


# ========== ABSTRACT BASE CLASSES ==========

class Shape(ABC):
    """Abstract base class for shapes."""
    
    def __init__(self, name: str):
        """Initialize shape with name."""
        self.name = name
    
    @abstractmethod
    def area(self) -> float:
        """Calculate area (must be implemented by subclasses)."""
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        """Calculate perimeter (must be implemented by subclasses)."""
        pass
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name} - Area: {self.area():.2f}, Perimeter: {self.perimeter():.2f}"
    
    def describe(self) -> str:
        """Describe the shape."""
        return f"This is a {self.name}"


class Rectangle(Shape):
    """Rectangle implementation of Shape."""
    
    def __init__(self, width: float, height: float):
        """Initialize rectangle."""
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def area(self) -> float:
        """Calculate rectangle area."""
        return self.width * self.height
    
    def perimeter(self) -> float:
        """Calculate rectangle perimeter."""
        return 2 * (self.width + self.height)
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"Rectangle(width={self.width}, height={self.height})"


class Circle(Shape):
    """Circle implementation of Shape."""
    
    def __init__(self, radius: float):
        """Initialize circle."""
        # Call the parent class (Shape) constructor with name "Circle"
        # This initializes the name attribute inherited from Shape
        super().__init__("Circle")
        self.radius = radius
    
    def area(self) -> float:
        """Calculate circle area."""
        return math.pi * self.radius ** 2
    
    def perimeter(self) -> float:
        """Calculate circle perimeter (circumference)."""
        return 2 * math.pi * self.radius
    
    # def __repr__(self) -> str:
    #     """Developer representation."""
    #     return f"Circle(radius={self.radius})"


class Triangle(Shape):
    """Triangle implementation of Shape."""
    
    def __init__(self, side_a: float, side_b: float, side_c: float):
        """Initialize triangle."""
        super().__init__("Triangle")
        self.side_a = side_a
        self.side_b = side_b
        self.side_c = side_c
        
        # Validate triangle inequality
        if not (side_a + side_b > side_c and 
                side_b + side_c > side_a and 
                side_c + side_a > side_b):
            raise ValueError("Invalid triangle: sides don't satisfy triangle inequality")
    
    def area(self) -> float:
        """Calculate triangle area using Heron's formula."""
        s = self.perimeter() / 2  # Semi-perimeter
        return math.sqrt(s * (s - self.side_a) * (s - self.side_b) * (s - self.side_c))
    
    def perimeter(self) -> float:
        """Calculate triangle perimeter."""
        return self.side_a + self.side_b + self.side_c
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"Triangle(side_a={self.side_a}, side_b={self.side_b}, side_c={self.side_c})"


# ========== PROPERTY DECORATORS ==========

class Temperature:
    """Temperature class demonstrating property decorators."""
    
    def __init__(self, celsius: float = 0):
        """Initialize temperature in Celsius."""
        self._celsius = celsius
    
    @property
    def celsius(self) -> float:
        """Get temperature in Celsius."""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value: float):
        """Set temperature in Celsius."""
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self) -> float:
        """Get temperature in Fahrenheit."""
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value: float):
        """Set temperature in Fahrenheit."""
        self.celsius = (value - 32) * 5/9
    
    @property
    def kelvin(self) -> float:
        """Get temperature in Kelvin."""
        return self._celsius + 273.15
    
    @kelvin.setter
    def kelvin(self, value: float):
        """Set temperature in Kelvin."""
        if value < 0:
            raise ValueError("Kelvin temperature cannot be negative")
        self.celsius = value - 273.15
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.celsius:.1f}°C ({self.fahrenheit:.1f}°F, {self.kelvin:.1f}K)"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"Temperature(celsius={self.celsius})"


# ========== CLASS METHODS AND STATIC METHODS ==========

class Person:
    """Person class demonstrating class and static methods."""
    
    population = 0  # Class variable
    
    def __init__(self, name: str, age: int):
        """Initialize person."""
        self.name = name
        self.age = age
        Person.population += 1
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name}, {self.age} years old"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"Person(name='{self.name}', age={self.age})"
    
    def __del__(self):
        """Destructor."""
        Person.population -= 1
    
    def greet(self) -> str:
        """Instance method."""
        return f"Hello, I'm {self.name}"
    
    @classmethod
    def get_population(cls) -> int:
        """Class method to get population."""
        return cls.population
    
    @classmethod
    def from_birth_year(cls, name: str, birth_year: int):
        """Alternative constructor using birth year."""
        import datetime
        current_year = datetime.datetime.now().year
        age = current_year - birth_year
        return cls(name, age)
    
    @staticmethod
    def is_adult(age: int) -> bool:
        """Static method to check if age represents an adult."""
        return age >= 18
    
    @staticmethod
    def calculate_age_difference(person1: 'Person', person2: 'Person') -> int:
        """Static method to calculate age difference."""
        return abs(person1.age - person2.age)


# ========== COMPOSITION EXAMPLE ==========

class Engine:
    """Engine class for composition example."""
    
    def __init__(self, horsepower: int, fuel_type: str):
        """Initialize engine."""
        self.horsepower = horsepower
        self.fuel_type = fuel_type
        self.running = False
    
    def start(self) -> str:
        """Start the engine."""
        self.running = True
        return f"Engine started ({self.horsepower} HP, {self.fuel_type})"
    
    def stop(self) -> str:
        """Stop the engine."""
        self.running = False
        return "Engine stopped"
    
    def __str__(self) -> str:
        """String representation."""
        status = "running" if self.running else "stopped"
        return f"{self.horsepower} HP {self.fuel_type} engine ({status})"


class Car:
    """Car class demonstrating composition."""
    
    def __init__(self, make: str, model: str, year: int, engine: Engine):
        """Initialize car with composition."""
        self.make = make
        self.model = model
        self.year = year
        self.engine = engine  # Composition: Car HAS-A Engine
        self.speed = 0
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.year} {self.make} {self.model}"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"Car(make='{self.make}', model='{self.model}', year={self.year}, engine={self.engine})"
    
    def start(self) -> str:
        """Start the car."""
        return f"Starting {self}: {self.engine.start()}"
    
    def stop(self) -> str:
        """Stop the car."""
        self.speed = 0
        return f"Stopping {self}: {self.engine.stop()}"
    
    def accelerate(self, amount: int) -> str:
        """Accelerate the car."""
        if not self.engine.running:
            return "Cannot accelerate: engine is not running"
        
        self.speed += amount
        return f"{self} accelerating to {self.speed} mph"
    
    def brake(self, amount: int) -> str:
        """Brake the car."""
        self.speed = max(0, self.speed - amount)
        return f"{self} braking to {self.speed} mph"


# ========== POLYMORPHISM DEMONSTRATION ==========

def demonstrate_polymorphism():
    """Demonstrate polymorphism with different objects."""
    print("=== Polymorphism Demonstration ===\n")
    
    # Create different animals
    animals = [
        Dog("Buddy", "Golden Retriever"),
        Cat("Whiskers", "Persian"),
        Eagle("Soar"),
        Duck("Quackers"),
        Penguin("Waddles")
    ]
    
    print("Animal sounds (polymorphism):")
    for animal in animals:
        print(f"  {animal.speak()}")
    
    print(f"\nAnimal movements (polymorphism):")
    for animal in animals:
        print(f"  {animal.move()}")
    
    # Shape polymorphism
    print(f"\nShape calculations (polymorphism):")
    shapes = [
        Rectangle(5, 3),
        Circle(4),
        Triangle(3, 4, 5)
    ]
    
    for shape in shapes:
        print(f"  {shape}")
    
    # Duck typing example
    print(f"\nDuck typing (if it walks like a duck...):")
    swimmers = [Duck("Donald"), Penguin("Pingu")]
    
    for swimmer in swimmers:
        if hasattr(swimmer, 'swim'):
            print(f"  {swimmer.swim()}")


def hackerrank_oop_problems():
    """
    Collection of HackerRank-style problems for OOP concepts.
    """
    
    def problem_1_bank_account():
        """
        Problem 1: Bank Account System
        
        Create a BankAccount class with:
        - Private balance attribute
        - Methods: deposit, withdraw, get_balance
        - Transaction history
        - Overdraft protection
        """
        
        class BankAccount:
            def __init__(self, account_number: str, initial_balance: float = 0):
                self.account_number = account_number
                self._balance = initial_balance
                self._transactions = []
            
            def deposit(self, amount: float) -> str:
                if amount <= 0:
                    return "Invalid deposit amount"
                
                self._balance += amount
                self._transactions.append(f"Deposit: +${amount:.2f}")
                return f"Deposited ${amount:.2f}. New balance: ${self._balance:.2f}"
            
            def withdraw(self, amount: float) -> str:
                if amount <= 0:
                    return "Invalid withdrawal amount"
                
                if amount > self._balance:
                    return "Insufficient funds"
                
                self._balance -= amount
                self._transactions.append(f"Withdrawal: -${amount:.2f}")
                return f"Withdrew ${amount:.2f}. New balance: ${self._balance:.2f}"
            
            def get_balance(self) -> float:
                return self._balance
            
            def get_transaction_history(self) -> List[str]:
                return self._transactions.copy()
            
            def __str__(self) -> str:
                return f"Account {self.account_number}: ${self._balance:.2f}"
        
        # Test the bank account
        account = BankAccount("12345", 100)
        
        print("Problem 1 - Bank Account System:")
        print(f"  Initial: {account}")
        print(f"  {account.deposit(50)}")
        print(f"  {account.withdraw(30)}")
        print(f"  {account.withdraw(200)}")  # Should fail
        print(f"  Transaction history: {account.get_transaction_history()}")
        
        return account
    
    def problem_2_library_system():
        """
        Problem 2: Library Management System
        
        Create classes for Book, Member, and Library with:
        - Book: title, author, ISBN, availability
        - Member: name, member_id, borrowed_books
        - Library: books collection, members, borrow/return methods
        """
        
        class Book:
            def __init__(self, title: str, author: str, isbn: str):
                self.title = title
                self.author = author
                self.isbn = isbn
                self.is_available = True
                self.borrowed_by = None
            
            def __str__(self) -> str:
                status = "Available" if self.is_available else f"Borrowed by {self.borrowed_by}"
                return f"'{self.title}' by {self.author} ({status})"
            
            def __eq__(self, other) -> bool:
                return isinstance(other, Book) and self.isbn == other.isbn
        
        class Member:
            def __init__(self, name: str, member_id: str):
                self.name = name
                self.member_id = member_id
                self.borrowed_books = []
            
            def __str__(self) -> str:
                return f"Member {self.name} (ID: {self.member_id})"
        
        class Library:
            def __init__(self, name: str):
                self.name = name
                self.books = []
                self.members = []
            
            def add_book(self, book: Book):
                self.books.append(book)
            
            def add_member(self, member: Member):
                self.members.append(member)
            
            def borrow_book(self, member_id: str, isbn: str) -> str:
                # Find member
                member = None
                for m in self.members:
                    if m.member_id == member_id:
                        member = m
                        break
                
                if not member:
                    return "Member not found"
                
                # Find book
                book = None
                for b in self.books:
                    if b.isbn == isbn:
                        book = b
                        break
                
                if not book:
                    return "Book not found"
                
                if not book.is_available:
                    return "Book is not available"
                
                # Borrow the book
                book.is_available = False
                book.borrowed_by = member.name
                member.borrowed_books.append(book)
                
                return f"{member.name} borrowed '{book.title}'"
            
            def return_book(self, member_id: str, isbn: str) -> str:
                # Find member
                member = None
                for m in self.members:
                    if m.member_id == member_id:
                        member = m
                        break
                
                if not member:
                    return "Member not found"
                
                # Find book in member's borrowed books
                book = None
                for b in member.borrowed_books:
                    if b.isbn == isbn:
                        book = b
                        break
                
                if not book:
                    return "Book not borrowed by this member"
                
                # Return the book
                book.is_available = True
                book.borrowed_by = None
                member.borrowed_books.remove(book)
                
                return f"{member.name} returned '{book.title}'"
        
        # Test the library system
        library = Library("City Library")
        
        # Add books
        book1 = Book("1984", "George Orwell", "978-0-452-28423-4")
        book2 = Book("To Kill a Mockingbird", "Harper Lee", "978-0-06-112008-4")
        library.add_book(book1)
        library.add_book(book2)
        
        # Add members
        member1 = Member("Alice Johnson", "M001")
        member2 = Member("Bob Smith", "M002")
        library.add_member(member1)
        library.add_member(member2)
        
        print("Problem 2 - Library Management System:")
        print(f"  {library.borrow_book('M001', '978-0-452-28423-4')}")
        print(f"  {library.borrow_book('M002', '978-0-452-28423-4')}")  # Should fail
        print(f"  {library.return_book('M001', '978-0-452-28423-4')}")
        print(f"  Books status:")
        for book in library.books:
            print(f"    {book}")
        
        return library
    
    def problem_3_vehicle_hierarchy():
        """
        Problem 3: Vehicle Hierarchy
        
        Create a vehicle hierarchy with:
        - Base Vehicle class
        - Car, Motorcycle, Truck subclasses
        - Different fuel efficiency calculations
        - Polymorphic behavior
        """
        
        class Vehicle(ABC):
            def __init__(self, make: str, model: str, year: int):
                self.make = make
                self.model = model
                self.year = year
                self.fuel_level = 100  # Percentage
            
            @abstractmethod
            def fuel_efficiency(self) -> float:
                """Miles per gallon."""
                pass
            
            @abstractmethod
            def max_speed(self) -> int:
                """Maximum speed in mph."""
                pass
            
            def drive(self, miles: float) -> str:
                fuel_needed = miles / self.fuel_efficiency()
                fuel_needed_percent = (fuel_needed / 20) * 100  # Assume 20 gallon tank
                
                if fuel_needed_percent > self.fuel_level:
                    return "Not enough fuel for this trip"
                
                self.fuel_level -= fuel_needed_percent
                return f"Drove {miles} miles. Fuel level: {self.fuel_level:.1f}%"
            
            def refuel(self) -> str:
                self.fuel_level = 100
                return "Refueled to 100%"
            
            def __str__(self) -> str:
                return f"{self.year} {self.make} {self.model}"
        
        class Car(Vehicle):
            def __init__(self, make: str, model: str, year: int, doors: int = 4):
                super().__init__(make, model, year)
                self.doors = doors
            
            def fuel_efficiency(self) -> float:
                return 30.0  # 30 MPG
            
            def max_speed(self) -> int:
                return 120  # 120 MPH
        
        class Motorcycle(Vehicle):
            def __init__(self, make: str, model: str, year: int, engine_size: int):
                super().__init__(make, model, year)
                self.engine_size = engine_size  # CC
            
            def fuel_efficiency(self) -> float:
                return 50.0  # 50 MPG
            
            def max_speed(self) -> int:
                return 180  # 180 MPH
        
        class Truck(Vehicle):
            def __init__(self, make: str, model: str, year: int, payload_capacity: int):
                super().__init__(make, model, year)
                self.payload_capacity = payload_capacity  # Pounds
            
            def fuel_efficiency(self) -> float:
                return 15.0  # 15 MPG
            
            def max_speed(self) -> int:
                return 85  # 85 MPH
        
        # Test the vehicle hierarchy
        vehicles = [
            Car("Toyota", "Camry", 2020),
            Motorcycle("Harley-Davidson", "Street 750", 2019, 750),
            Truck("Ford", "F-150", 2021, 3000)
        ]
        
        print("Problem 3 - Vehicle Hierarchy:")
        for vehicle in vehicles:
            print(f"  {vehicle}:")
            print(f"    Fuel efficiency: {vehicle.fuel_efficiency()} MPG")
            print(f"    Max speed: {vehicle.max_speed()} MPH")
            print(f"    {vehicle.drive(100)}")
        
        return vehicles
    
    # Run all problems
    print("=== HackerRank Style OOP Problems ===\n")
    problem_1_bank_account()
    print()
    problem_2_library_system()
    print()
    problem_3_vehicle_hierarchy()


if __name__ == "__main__":
    # Demonstrate basic magic methods
    print("=== Magic Methods Demonstration ===\n")
    
    # Point class
    p1 = Point(3, 4)
    p2 = Point(1, 2)
    
    print(f"Point operations:")
    print(f"  p1 = {p1}")
    print(f"  p2 = {p2}")
    print(f"  p1 + p2 = {p1 + p2}")
    print(f"  p1 - p2 = {p1 - p2}")
    print(f"  p1 * 2 = {p1 * 2}")
    print(f"  abs(p1) = {abs(p1)}")
    print(f"  p1 == p2: {p1 == p2}")
    print(f"  p1 < p2: {p1 < p2}")
    
    # Vector class
    v1 = Vector(1, 2, 3)
    v2 = Vector(4, 5, 6)
    
    print(f"\nVector operations:")
    print(f"  v1 = {v1}")
    print(f"  v2 = {v2}")
    print(f"  len(v1) = {len(v1)}")
    print(f"  v1[1] = {v1[1]}")
    print(f"  v1 + v2 = {v1 + v2}")
    print(f"  v1 * 2 = {v1 * 2}")
    print(f"  v1 • v2 = {v1 * v2}")  # Dot product
    print(f"  |v1| = {abs(v1)}")
    print(f"  2 in v1: {2 in v1}")
    
    # Matrix class
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])
    
    print(f"\nMatrix operations:")
    print(f"  m1 =\n{m1}")
    print(f"  m2 =\n{m2}")
    print(f"  m1 + m2 =\n{m1 + m2}")
    print(f"  m1 * 2 =\n{m1 * 2}")
    
    print("\n" + "="*60 + "\n")
    
    # Demonstrate inheritance and polymorphism
    demonstrate_polymorphism()
    
    print("\n" + "="*60 + "\n")
    
    # Demonstrate properties
    print("=== Property Decorators ===\n")
    temp = Temperature(25)
    print(f"Temperature: {temp}")
    temp.fahrenheit = 100
    print(f"After setting to 100°F: {temp}")
    
    print("\n" + "="*60 + "\n")
    
    # Demonstrate class methods and static methods
    print("=== Class and Static Methods ===\n")
    person1 = Person("Alice", 30)
    person2 = Person.from_birth_year("Bob", 1990)
    
    print(f"Person 1: {person1}")
    print(f"Person 2: {person2}")
    print(f"Population: {Person.get_population()}")
    print(f"Is Alice an adult? {Person.is_adult(person1.age)}")
    print(f"Age difference: {Person.calculate_age_difference(person1, person2)} years")
    
    print("\n" + "="*60 + "\n")
    
    # Demonstrate composition
    print("=== Composition ===\n")
    engine = Engine(300, "Gasoline")
    car = Car("Tesla", "Model S", 2021, engine)
    
    print(f"Car: {car}")
    print(f"Engine: {car.engine}")
    print(f"{car.start()}")
    print(f"{car.accelerate(30)}")
    print(f"{car.brake(10)}")
    print(f"{car.stop()}")
    
    print("\n" + "="*60 + "\n")
    
    # Run HackerRank-style problems
    hackerrank_oop_problems()
