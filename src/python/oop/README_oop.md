# Object-Oriented Programming in Python

This module provides comprehensive coverage of Object-Oriented Programming concepts in Python, focusing on topics commonly tested in data science interviews and coding assessments.

## Overview

Object-Oriented Programming (OOP) is a fundamental paradigm in Python and essential for data science work. This module covers everything from basic classes to advanced design patterns, with emphasis on practical applications and interview-style problems.

## Key Concepts Covered

### 1. Classes and Objects
- **Class definition**: Creating blueprints for objects
- **Object instantiation**: Creating instances from classes
- **Attributes**: Instance and class variables
- **Methods**: Instance, class, and static methods
- **Constructors**: `__init__` method and object initialization

### 2. Magic Methods (Dunder Methods)
- **String representation**: `__str__`, `__repr__`
- **Arithmetic operations**: `__add__`, `__sub__`, `__mul__`, `__div__`
- **Comparison operations**: `__eq__`, `__lt__`, `__gt__`, `__le__`, `__ge__`
- **Container methods**: `__len__`, `__getitem__`, `__setitem__`, `__contains__`
- **Iteration protocol**: `__iter__`, `__next__`
- **Context managers**: `__enter__`, `__exit__`
- **Attribute access**: `__getattr__`, `__setattr__`, `__delattr__`
- **Callable objects**: `__call__`
- **Hashing**: `__hash__`

### 3. Inheritance
- **Single inheritance**: One parent class
- **Multiple inheritance**: Multiple parent classes
- **Method Resolution Order (MRO)**: Diamond problem resolution
- **Super() function**: Calling parent methods
- **Method overriding**: Redefining parent methods
- **Abstract base classes**: Enforcing interface contracts

### 4. Polymorphism
- **Method overriding**: Same method name, different behavior
- **Duck typing**: "If it walks like a duck..."
- **Interface polymorphism**: Common interface, different implementations
- **Operator overloading**: Custom behavior for operators

### 5. Encapsulation
- **Public attributes**: Accessible from anywhere
- **Protected attributes**: Convention with single underscore `_`
- **Private attributes**: Name mangling with double underscore `__`
- **Property decorators**: Controlled attribute access
- **Getters and setters**: Custom attribute behavior

### 6. Advanced OOP Concepts
- **Descriptors**: Custom attribute access control
- **Metaclasses**: Classes that create classes
- **Mixins**: Reusable functionality through multiple inheritance
- **Composition vs Inheritance**: "Has-a" vs "Is-a" relationships
- **Design patterns**: Common solutions to recurring problems

## Files Structure

```
src/oop/
├── python_oop.py           # Core OOP demonstrations and concepts
├── practice_exercises.py   # Additional practice problems and patterns
└── README_oop.md          # This documentation
```

## Quick Start

### Basic Usage

```python
from python_oop import Point, Vector, Matrix, Animal, Dog, Cat

# Basic magic methods
p1 = Point(3, 4)
p2 = Point(1, 2)
print(f"p1 + p2 = {p1 + p2}")
print(f"Distance: {p1.distance_to(p2)}")

# Vector operations
v1 = Vector(1, 2, 3)
v2 = Vector(4, 5, 6)
print(f"Dot product: {v1 * v2}")
print(f"Magnitude: {abs(v1)}")

# Inheritance and polymorphism
animals = [Dog("Buddy"), Cat("Whiskers")]
for animal in animals:
    print(animal.speak())  # Polymorphic behavior
```

### Advanced Patterns

```python
from practice_exercises import OOPExercises

exercises = OOPExercises()

# Custom iterator
seq = exercises.NumberSequence(1, 10, 2)
print(list(seq))  # [1, 3, 5, 7, 9]

# Observer pattern
stock = exercises.StockPrice("AAPL", 150.00)
observer = exercises.StockObserver("Investor")
stock.add_observer(observer)
stock.price = 155.50  # Notifies observer

# Command pattern with undo/redo
calc = exercises.Calculator()
calc.execute_command(exercises.AddCommand(calc, 10))
calc.undo()  # Undoes the addition
```

### HackerRank-Style Problems

```python
from python_oop import hackerrank_oop_problems

# Run practice problems
hackerrank_oop_problems()
```

## Magic Methods Reference

### String Representation
```python
class MyClass:
    def __str__(self):
        return "User-friendly string"
    
    def __repr__(self):
        return "MyClass(developer_info)"
```

### Arithmetic Operations
```python
class Number:
    def __init__(self, value):
        self.value = value
    
    def __add__(self, other):
        return Number(self.value + other.value)
    
    def __mul__(self, scalar):
        return Number(self.value * scalar)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
```

### Comparison Operations
```python
class Comparable:
    def __eq__(self, other):
        return self.value == other.value
    
    def __lt__(self, other):
        return self.value < other.value
    
    def __le__(self, other):
        return self < other or self == other
```

### Container Protocol
```python
class Container:
    def __init__(self):
        self.items = []
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
    
    def __setitem__(self, index, value):
        self.items[index] = value
    
    def __contains__(self, item):
        return item in self.items
```

### Iterator Protocol
```python
class Iterator:
    def __init__(self, data):
        self.data = data
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        result = self.data[self.index]
        self.index += 1
        return result
```

## Inheritance Patterns

### Single Inheritance
```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return f"{self.name} barks"
```

### Multiple Inheritance
```python
class Swimmer:
    def swim(self):
        return "Swimming"

class Flyer:
    def fly(self):
        return "Flying"

class Duck(Animal, Swimmer, Flyer):
    def speak(self):
        return f"{self.name} quacks"
```

### Abstract Base Classes
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)
```

## Property Decorators

### Basic Properties
```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9
```

### Computed Properties
```python
class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    @property
    def area(self):
        return 3.14159 * self.radius ** 2
    
    @property
    def circumference(self):
        return 2 * 3.14159 * self.radius
```

## Class and Static Methods

### Class Methods
```python
class Person:
    population = 0
    
    def __init__(self, name):
        self.name = name
        Person.population += 1
    
    @classmethod
    def get_population(cls):
        return cls.population
    
    @classmethod
    def from_string(cls, name_str):
        name = name_str.split('-')[0]
        return cls(name)
```

### Static Methods
```python
class MathUtils:
    @staticmethod
    def add(a, b):
        return a + b
    
    @staticmethod
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
```

## Design Patterns

### Singleton Pattern
```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### Factory Pattern
```python
class ShapeFactory:
    @staticmethod
    def create_shape(shape_type, **kwargs):
        if shape_type == 'circle':
            return Circle(kwargs['radius'])
        elif shape_type == 'rectangle':
            return Rectangle(kwargs['width'], kwargs['height'])
        else:
            raise ValueError(f"Unknown shape: {shape_type}")
```

### Observer Pattern
```python
class Observable:
    def __init__(self):
        self._observers = []
    
    def add_observer(self, observer):
        self._observers.append(observer)
    
    def notify_observers(self, *args, **kwargs):
        for observer in self._observers:
            observer.update(self, *args, **kwargs)

class Observer:
    def update(self, observable, *args, **kwargs):
        print(f"Observer notified: {args}, {kwargs}")
```

## Common Interview Questions

### 1. "What are magic methods and why are they useful?"

**Answer**:
Magic methods (dunder methods) allow you to define how objects behave with built-in operations:

```python
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"

# Usage
v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2  # Uses __add__
print(v3)     # Uses __str__
```

### 2. "Explain the difference between `__str__` and `__repr__`"

**Answer**:
- `__str__`: User-friendly string representation
- `__repr__`: Developer-friendly, unambiguous representation

```python
class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def __str__(self):
        return f"Point at ({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

p = Point(1, 2)
print(str(p))   # "Point at (1, 2)"
print(repr(p))  # "Point(x=1, y=2)"
```

### 3. "What is Method Resolution Order (MRO)?"

**Answer**:
MRO determines the order in which methods are resolved in multiple inheritance:

```python
class A:
    def method(self):
        print("A")

class B(A):
    def method(self):
        print("B")

class C(A):
    def method(self):
        print("C")

class D(B, C):
    pass

print(D.__mro__)  # Shows resolution order
D().method()      # Calls B.method() due to MRO
```

### 4. "When would you use composition over inheritance?"

**Answer**:
Use composition when you have a "has-a" relationship rather than "is-a":

```python
# Inheritance (is-a): Car IS-A Vehicle
class Vehicle:
    pass

class Car(Vehicle):
    pass

# Composition (has-a): Car HAS-A Engine
class Engine:
    def start(self):
        return "Engine started"

class Car:
    def __init__(self):
        self.engine = Engine()  # Composition
    
    def start(self):
        return self.engine.start()
```

### 5. "How do you implement a custom iterator?"

**Answer**:
Implement `__iter__` and `__next__` methods:

```python
class Counter:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start >= self.end:
            raise StopIteration
        self.start += 1
        return self.start - 1

# Usage
for num in Counter(1, 5):
    print(num)  # Prints 1, 2, 3, 4
```

## Best Practices

### Code Organization
1. **Single Responsibility**: Each class should have one reason to change
2. **Open/Closed Principle**: Open for extension, closed for modification
3. **Liskov Substitution**: Subclasses should be substitutable for base classes
4. **Interface Segregation**: Many specific interfaces are better than one general
5. **Dependency Inversion**: Depend on abstractions, not concretions

### Naming Conventions
```python
class MyClass:              # PascalCase for classes
    class_variable = 0      # snake_case for variables
    
    def __init__(self):
        self.public_attr = 1        # Public attribute
        self._protected_attr = 2    # Protected (convention)
        self.__private_attr = 3     # Private (name mangling)
    
    def public_method(self):        # Public method
        pass
    
    def _protected_method(self):    # Protected method
        pass
    
    def __private_method(self):     # Private method
        pass
```

### Error Handling
```python
class CustomError(Exception):
    """Custom exception class."""
    pass

class MyClass:
    def method(self, value):
        if not isinstance(value, int):
            raise TypeError("Value must be an integer")
        if value < 0:
            raise ValueError("Value must be non-negative")
        if value > 100:
            raise CustomError("Value exceeds maximum limit")
```

## Performance Considerations

### Memory Usage
```python
# Use __slots__ to reduce memory usage
class Point:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Regular class uses more memory due to __dict__
class RegularPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

### Method Caching
```python
from functools import lru_cache

class ExpensiveCalculation:
    @lru_cache(maxsize=128)
    def fibonacci(self, n):
        if n < 2:
            return n
        return self.fibonacci(n-1) + self.fibonacci(n-2)
```

## Advanced Topics

### Metaclasses
```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(metaclass=SingletonMeta):
    pass
```

### Descriptors
```python
class ValidatedAttribute:
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be <= {self.max_value}")
        obj.__dict__[self.name] = value

class Product:
    price = ValidatedAttribute(min_value=0)
    rating = ValidatedAttribute(min_value=0, max_value=5)
```

## Practice Problem Categories

### 1. Basic OOP (5 problems)
- Class definition and instantiation
- Magic methods implementation
- Property decorators
- Class and static methods
- Basic inheritance

### 2. Advanced Inheritance (4 problems)
- Multiple inheritance and MRO
- Abstract base classes
- Method overriding
- Super() usage

### 3. Design Patterns (6 problems)
- Singleton pattern
- Factory pattern
- Observer pattern
- Command pattern
- Strategy pattern
- Decorator pattern

### 4. Magic Methods (8 problems)
- Arithmetic operations
- Comparison operations
- Container protocol
- Iterator protocol
- Context managers
- String representation

### 5. Real-world Applications (7 problems)
- Bank account system
- Library management
- Vehicle hierarchy
- Shape calculations
- Data structures
- Game development
- API design

## Interview Preparation Tips

### Study Strategy
1. **Master the basics**: Classes, objects, inheritance
2. **Practice magic methods**: Understand when and how to use them
3. **Learn design patterns**: Common solutions to recurring problems
4. **Implement data structures**: Stack, queue, linked list using OOP
5. **Solve real-world problems**: Bank systems, game objects, etc.

### Common Mistakes to Avoid
1. **Overusing inheritance**: Prefer composition when appropriate
2. **Not implementing `__repr__`**: Always provide developer-friendly representation
3. **Ignoring MRO**: Understand method resolution in multiple inheritance
4. **Misusing private attributes**: Use single underscore for protected
5. **Not handling exceptions**: Validate inputs and handle edge cases

### Problem-Solving Approach
1. **Identify relationships**: Is-a (inheritance) vs Has-a (composition)
2. **Define interfaces**: What methods should be public?
3. **Consider extensibility**: How might this class be extended?
4. **Think about edge cases**: What could go wrong?
5. **Test polymorphism**: Can objects be used interchangeably?

## Resources for Further Learning

### Documentation
- [Python Data Model](https://docs.python.org/3/reference/datamodel.html)
- [Python Classes Tutorial](https://docs.python.org/3/tutorial/classes.html)
- [ABC Module](https://docs.python.org/3/library/abc.html)

### Books
- "Effective Python" by Brett Slatkin
- "Python Tricks" by Dan Bader
- "Design Patterns" by Gang of Four
- "Clean Code" by Robert Martin

### Online Practice
- **LeetCode**: OOP design problems
- **HackerRank**: Object-oriented programming challenges
- **Codewars**: Class design katas

## Contributing

To add new exercises or improve existing ones:

1. Add new classes with comprehensive docstrings
2. Include practical examples and use cases
3. Provide test cases and expected outputs
4. Update this documentation with new patterns

## License

This educational material is provided under the MIT License for interview preparation purposes.
