"""
Practice Exercises for Object-Oriented Programming - Interview Preparation

This module contains additional practice exercises focusing on OOP concepts
commonly tested in data science interviews and coding assessments.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import datetime
import json
from functools import wraps


class OOPExercises:
    """
    Additional practice exercises for Object-Oriented Programming concepts.
    """
    
    def __init__(self):
        """Initialize the exercises class."""
        pass
    
    # ========== EXERCISE 1: CUSTOM ITERATOR ==========
    
    class NumberSequence:
        """
        Custom iterator that generates a sequence of numbers.
        Demonstrates __iter__ and __next__ magic methods.
        """
        
        def __init__(self, start: int, end: int, step: int = 1):
            """Initialize the sequence."""
            self.start = start
            self.end = end
            self.step = step
            self.current = start
        
        def __iter__(self):
            """Return the iterator object."""
            return self
        
        def __next__(self):
            """Return the next item in the sequence."""
            if (self.step > 0 and self.current >= self.end) or \
               (self.step < 0 and self.current <= self.end):
                raise StopIteration
            
            result = self.current
            self.current += self.step
            return result
        
        def __repr__(self):
            """String representation."""
            return f"NumberSequence(start={self.start}, end={self.end}, step={self.step})"
    
    # ========== EXERCISE 2: CONTEXT MANAGER ==========
    
    class FileManager:
        """
        Custom context manager for file operations.
        Demonstrates __enter__ and __exit__ magic methods.
        """
        
        def __init__(self, filename: str, mode: str = 'r'):
            """Initialize file manager."""
            self.filename = filename
            self.mode = mode
            self.file = None
        
        def __enter__(self):
            """Enter the context."""
            print(f"Opening file: {self.filename}")
            self.file = open(self.filename, self.mode)
            return self.file
        
        def __exit__(self, exc_type, exc_value, traceback):
            """Exit the context."""
            if self.file:
                print(f"Closing file: {self.filename}")
                self.file.close()
            
            if exc_type:
                print(f"Exception occurred: {exc_type.__name__}: {exc_value}")
            
            return False  # Don't suppress exceptions
    
    # ========== EXERCISE 3: DESCRIPTOR PROTOCOL ==========
    
    class ValidatedAttribute:
        """
        Descriptor that validates attribute values.
        Demonstrates __get__, __set__, and __delete__ methods.
        """
        
        def __init__(self, min_value: float = None, max_value: float = None):
            """Initialize validator."""
            self.min_value = min_value
            self.max_value = max_value
            self.name = None
        
        def __set_name__(self, owner, name):
            """Called when the descriptor is assigned to a class attribute."""
            self.name = name
        
        def __get__(self, obj, objtype=None):
            """Get the attribute value."""
            if obj is None:
                return self
            return obj.__dict__.get(self.name)
        
        def __set__(self, obj, value):
            """Set the attribute value with validation."""
            if not isinstance(value, (int, float)):
                raise TypeError(f"{self.name} must be a number")
            
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"{self.name} must be >= {self.min_value}")
            
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"{self.name} must be <= {self.max_value}")
            
            obj.__dict__[self.name] = value
        
        def __delete__(self, obj):
            """Delete the attribute."""
            if self.name in obj.__dict__:
                del obj.__dict__[self.name]
    
    class Product:
        """Product class using validated attributes."""
        
        price = ValidatedAttribute(min_value=0)
        rating = ValidatedAttribute(min_value=0, max_value=5)
        
        def __init__(self, name: str, price: float, rating: float = 0):
            """Initialize product."""
            self.name = name
            self.price = price
            self.rating = rating
        
        def __str__(self):
            """String representation."""
            return f"{self.name}: ${self.price:.2f} (Rating: {self.rating}/5)"
    
    # ========== EXERCISE 4: METACLASS EXAMPLE ==========
    
    class SingletonMeta(type):
        """
        Metaclass that creates singleton instances.
        """
        
        _instances = {}
        
        def __call__(cls, *args, **kwargs):
            """Control instance creation."""
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]
    
    class DatabaseConnection(metaclass=SingletonMeta):
        """
        Database connection using singleton pattern.
        """
        
        def __init__(self, host: str = "localhost", port: int = 5432):
            """Initialize connection."""
            if not hasattr(self, 'initialized'):
                self.host = host
                self.port = port
                self.connected = False
                self.initialized = True
        
        def connect(self) -> str:
            """Connect to database."""
            self.connected = True
            return f"Connected to {self.host}:{self.port}"
        
        def disconnect(self) -> str:
            """Disconnect from database."""
            self.connected = False
            return "Disconnected from database"
        
        def __str__(self):
            """String representation."""
            status = "connected" if self.connected else "disconnected"
            return f"DatabaseConnection({self.host}:{self.port}, {status})"
    
    # ========== EXERCISE 5: DECORATOR CLASS ==========
    
    class TimedMethod:
        """
        Decorator class that times method execution.
        """
        
        def __init__(self, func):
            """Initialize decorator."""
            self.func = func
            self.call_count = 0
            self.total_time = 0
            wraps(func)(self)
        
        def __call__(self, *args, **kwargs):
            """Execute the decorated function."""
            import time
            
            start_time = time.time()
            result = self.func(*args, **kwargs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            self.call_count += 1
            self.total_time += execution_time
            
            print(f"{self.func.__name__} executed in {execution_time:.4f}s")
            return result
        
        def get_stats(self) -> Dict[str, float]:
            """Get execution statistics."""
            avg_time = self.total_time / self.call_count if self.call_count > 0 else 0
            return {
                'call_count': self.call_count,
                'total_time': self.total_time,
                'average_time': avg_time
            }
    
    # ========== EXERCISE 6: ADVANCED INHERITANCE ==========
    
    class Mixin:
        """Base mixin class."""
        pass
    
    class SerializableMixin(Mixin):
        """Mixin for JSON serialization."""
        
        def to_json(self) -> str:
            """Convert object to JSON string."""
            # Get all non-private attributes
            data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
            return json.dumps(data, default=str)
        
        @classmethod
        def from_json(cls, json_str: str):
            """Create object from JSON string."""
            data = json.loads(json_str)
            return cls(**data)
    
    class ComparableMixin(Mixin):
        """Mixin for comparison operations."""
        
        def __eq__(self, other):
            """Equality comparison."""
            if not isinstance(other, self.__class__):
                return NotImplemented
            return self.__dict__ == other.__dict__
        
        def __lt__(self, other):
            """Less than comparison (by string representation)."""
            if not isinstance(other, self.__class__):
                return NotImplemented
            return str(self) < str(other)
        
        def __le__(self, other):
            """Less than or equal comparison."""
            return self < other or self == other
        
        def __gt__(self, other):
            """Greater than comparison."""
            return not self <= other
        
        def __ge__(self, other):
            """Greater than or equal comparison."""
            return not self < other
        
        def __ne__(self, other):
            """Not equal comparison."""
            return not self == other
    
    class Student(SerializableMixin, ComparableMixin):
        """Student class with multiple mixins."""
        
        def __init__(self, name: str, student_id: str, gpa: float):
            """Initialize student."""
            self.name = name
            self.student_id = student_id
            self.gpa = gpa
        
        def __str__(self):
            """String representation."""
            return f"{self.name} (ID: {self.student_id}, GPA: {self.gpa})"
        
        def __repr__(self):
            """Developer representation."""
            return f"Student(name='{self.name}', student_id='{self.student_id}', gpa={self.gpa})"
    
    # ========== EXERCISE 7: FACTORY PATTERN ==========
    
    class ShapeFactory:
        """Factory class for creating shapes."""
        
        @staticmethod
        def create_shape(shape_type: str, **kwargs):
            """Create a shape based on type."""
            shape_type = shape_type.lower()
            
            if shape_type == 'circle':
                return Circle(kwargs.get('radius', 1))
            elif shape_type == 'rectangle':
                return Rectangle(kwargs.get('width', 1), kwargs.get('height', 1))
            elif shape_type == 'triangle':
                return Triangle(
                    kwargs.get('side_a', 1),
                    kwargs.get('side_b', 1),
                    kwargs.get('side_c', 1)
                )
            else:
                raise ValueError(f"Unknown shape type: {shape_type}")
    
    class Shape(ABC):
        """Abstract base class for shapes."""
        
        @abstractmethod
        def area(self) -> float:
            """Calculate area."""
            pass
        
        @abstractmethod
        def perimeter(self) -> float:
            """Calculate perimeter."""
            pass
    
    class Circle(Shape):
        """Circle implementation."""
        
        def __init__(self, radius: float):
            """Initialize circle."""
            self.radius = radius
        
        def area(self) -> float:
            """Calculate circle area."""
            import math
            return math.pi * self.radius ** 2
        
        def perimeter(self) -> float:
            """Calculate circle perimeter."""
            import math
            return 2 * math.pi * self.radius
        
        def __str__(self):
            """String representation."""
            return f"Circle(radius={self.radius})"
    
    class Rectangle(Shape):
        """Rectangle implementation."""
        
        def __init__(self, width: float, height: float):
            """Initialize rectangle."""
            self.width = width
            self.height = height
        
        def area(self) -> float:
            """Calculate rectangle area."""
            return self.width * self.height
        
        def perimeter(self) -> float:
            """Calculate rectangle perimeter."""
            return 2 * (self.width + self.height)
        
        def __str__(self):
            """String representation."""
            return f"Rectangle(width={self.width}, height={self.height})"
    
    class Triangle(Shape):
        """Triangle implementation."""
        
        def __init__(self, side_a: float, side_b: float, side_c: float):
            """Initialize triangle."""
            # Validate triangle inequality
            if not (side_a + side_b > side_c and 
                    side_b + side_c > side_a and 
                    side_c + side_a > side_b):
                raise ValueError("Invalid triangle: sides don't satisfy triangle inequality")
            
            self.side_a = side_a
            self.side_b = side_b
            self.side_c = side_c
        
        def area(self) -> float:
            """Calculate triangle area using Heron's formula."""
            import math
            s = self.perimeter() / 2
            return math.sqrt(s * (s - self.side_a) * (s - self.side_b) * (s - self.side_c))
        
        def perimeter(self) -> float:
            """Calculate triangle perimeter."""
            return self.side_a + self.side_b + self.side_c
        
        def __str__(self):
            """String representation."""
            return f"Triangle(sides={self.side_a}, {self.side_b}, {self.side_c})"
    
    # ========== EXERCISE 8: OBSERVER PATTERN ==========
    
    class Observable:
        """Observable class for observer pattern."""
        
        def __init__(self):
            """Initialize observable."""
            self._observers = []
        
        def add_observer(self, observer):
            """Add an observer."""
            if observer not in self._observers:
                self._observers.append(observer)
        
        def remove_observer(self, observer):
            """Remove an observer."""
            if observer in self._observers:
                self._observers.remove(observer)
        
        def notify_observers(self, *args, **kwargs):
            """Notify all observers."""
            for observer in self._observers:
                observer.update(self, *args, **kwargs)
    
    class StockPrice(Observable):
        """Stock price that notifies observers of changes."""
        
        def __init__(self, symbol: str, price: float):
            """Initialize stock price."""
            super().__init__()
            self.symbol = symbol
            self._price = price
        
        @property
        def price(self) -> float:
            """Get current price."""
            return self._price
        
        @price.setter
        def price(self, new_price: float):
            """Set new price and notify observers."""
            old_price = self._price
            self._price = new_price
            self.notify_observers(old_price=old_price, new_price=new_price)
        
        def __str__(self):
            """String representation."""
            return f"{self.symbol}: ${self.price:.2f}"
    
    class StockObserver:
        """Observer for stock price changes."""
        
        def __init__(self, name: str):
            """Initialize observer."""
            self.name = name
        
        def update(self, observable, *args, **kwargs):
            """Handle stock price update."""
            old_price = kwargs.get('old_price')
            new_price = kwargs.get('new_price')
            change = new_price - old_price
            direction = "↑" if change > 0 else "↓" if change < 0 else "→"
            
            print(f"{self.name} notified: {observable.symbol} {direction} "
                  f"${old_price:.2f} → ${new_price:.2f} (${change:+.2f})")
    
    # ========== EXERCISE 9: COMMAND PATTERN ==========
    
    class Command(ABC):
        """Abstract command interface."""
        
        @abstractmethod
        def execute(self):
            """Execute the command."""
            pass
        
        @abstractmethod
        def undo(self):
            """Undo the command."""
            pass
    
    class Calculator:
        """Calculator that supports undo/redo operations."""
        
        def __init__(self):
            """Initialize calculator."""
            self.value = 0
            self.history = []
            self.history_index = -1
        
        def execute_command(self, command):
            """Execute a command and add to history."""
            # Remove any commands after current position
            self.history = self.history[:self.history_index + 1]
            
            # Execute command
            command.execute()
            
            # Add to history
            self.history.append(command)
            self.history_index += 1
        
        def undo(self):
            """Undo the last command."""
            if self.history_index >= 0:
                command = self.history[self.history_index]
                command.undo()
                self.history_index -= 1
                return f"Undid: {command}"
            return "Nothing to undo"
        
        def redo(self):
            """Redo the next command."""
            if self.history_index < len(self.history) - 1:
                self.history_index += 1
                command = self.history[self.history_index]
                command.execute()
                return f"Redid: {command}"
            return "Nothing to redo"
        
        def __str__(self):
            """String representation."""
            return f"Calculator(value={self.value})"
    
    class AddCommand(Command):
        """Command to add a value."""
        
        def __init__(self, calculator, value):
            """Initialize add command."""
            self.calculator = calculator
            self.value = value
        
        def execute(self):
            """Execute addition."""
            self.calculator.value += self.value
        
        def undo(self):
            """Undo addition."""
            self.calculator.value -= self.value
        
        def __str__(self):
            """String representation."""
            return f"Add {self.value}"
    
    class MultiplyCommand(Command):
        """Command to multiply by a value."""
        
        def __init__(self, calculator, value):
            """Initialize multiply command."""
            self.calculator = calculator
            self.value = value
            self.previous_value = None
        
        def execute(self):
            """Execute multiplication."""
            self.previous_value = self.calculator.value
            self.calculator.value *= self.value
        
        def undo(self):
            """Undo multiplication."""
            self.calculator.value = self.previous_value
        
        def __str__(self):
            """String representation."""
            return f"Multiply by {self.value}"


def run_oop_exercises():
    """Run all OOP practice exercises."""
    exercises = OOPExercises()
    
    print("=== OOP Practice Exercises ===\n")
    
    # Exercise 1: Custom Iterator
    print("1. Custom Iterator:")
    seq = exercises.NumberSequence(1, 10, 2)
    print(f"   Sequence: {list(seq)}")
    
    # Exercise 2: Context Manager (create a temporary file for demo)
    print("\n2. Context Manager:")
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write("Hello, World!")
    
    try:
        with exercises.FileManager(temp_filename, 'r') as f:
            content = f.read()
            print(f"   File content: '{content}'")
    finally:
        os.unlink(temp_filename)
    
    # Exercise 3: Descriptor Protocol
    print("\n3. Descriptor Protocol:")
    try:
        product = exercises.Product("Laptop", 999.99, 4.5)
        print(f"   Product: {product}")
        
        product.price = 899.99
        print(f"   After price change: {product}")
        
        # This should raise an error
        try:
            product.rating = 6.0
        except ValueError as e:
            print(f"   Validation error: {e}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Exercise 4: Singleton Pattern
    print("\n4. Singleton Pattern:")
    db1 = exercises.DatabaseConnection("localhost", 5432)
    db2 = exercises.DatabaseConnection("remotehost", 3306)
    
    print(f"   db1: {db1}")
    print(f"   db2: {db2}")
    print(f"   Same instance? {db1 is db2}")
    
    # Exercise 5: Decorator Class
    print("\n5. Decorator Class:")
    
    class TestClass:
        @exercises.TimedMethod
        def slow_method(self):
            import time
            time.sleep(0.01)  # Simulate slow operation
            return "Done"
    
    test_obj = TestClass()
    result = test_obj.slow_method()
    stats = test_obj.slow_method.get_stats()
    print(f"   Method result: {result}")
    print(f"   Execution stats: {stats}")
    
    # Exercise 6: Multiple Inheritance with Mixins
    print("\n6. Multiple Inheritance with Mixins:")
    student1 = exercises.Student("Alice Johnson", "S001", 3.8)
    student2 = exercises.Student("Bob Smith", "S002", 3.6)
    
    print(f"   Student 1: {student1}")
    print(f"   Student 2: {student2}")
    print(f"   student1 > student2: {student1 > student2}")
    
    # JSON serialization
    json_str = student1.to_json()
    print(f"   JSON: {json_str}")
    
    # Exercise 7: Factory Pattern
    print("\n7. Factory Pattern:")
    shapes = [
        exercises.ShapeFactory.create_shape('circle', radius=5),
        exercises.ShapeFactory.create_shape('rectangle', width=4, height=3),
        exercises.ShapeFactory.create_shape('triangle', side_a=3, side_b=4, side_c=5)
    ]
    
    for shape in shapes:
        print(f"   {shape}: Area = {shape.area():.2f}, Perimeter = {shape.perimeter():.2f}")
    
    # Exercise 8: Observer Pattern
    print("\n8. Observer Pattern:")
    stock = exercises.StockPrice("AAPL", 150.00)
    
    observer1 = exercises.StockObserver("Investor 1")
    observer2 = exercises.StockObserver("Investor 2")
    
    stock.add_observer(observer1)
    stock.add_observer(observer2)
    
    print("   Price changes:")
    stock.price = 155.50
    stock.price = 148.75
    
    # Exercise 9: Command Pattern
    print("\n9. Command Pattern:")
    calc = exercises.Calculator()
    
    print(f"   Initial: {calc}")
    
    # Execute commands
    calc.execute_command(exercises.AddCommand(calc, 10))
    print(f"   After add 10: {calc}")
    
    calc.execute_command(exercises.MultiplyCommand(calc, 3))
    print(f"   After multiply by 3: {calc}")
    
    calc.execute_command(exercises.AddCommand(calc, 5))
    print(f"   After add 5: {calc}")
    
    # Undo operations
    print(f"   {calc.undo()}: {calc}")
    print(f"   {calc.undo()}: {calc}")
    
    # Redo operations
    print(f"   {calc.redo()}: {calc}")


if __name__ == "__main__":
    run_oop_exercises()
