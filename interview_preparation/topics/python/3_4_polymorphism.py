"""
Python Polymorphism - Interview Preparation
Topic 3.4: Polymorphism

This module covers:
- Duck Typing: "If it walks like a duck..."
- Method Overloading: Not directly supported, alternatives
- Operator Overloading: __add__, __sub__, etc.
"""

# ============================================================================
# 1. DUCK TYPING
# ============================================================================

print("=" * 70)
print("1. DUCK TYPING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Understanding Duck Typing
# ----------------------------------------------------------------------------
print("\n--- 1.1 Understanding Duck Typing ---")
print("""
DUCK TYPING PRINCIPLE:
"If it walks like a duck and quacks like a duck, it's a duck."

Python uses duck typing - objects are used based on their behavior
(what methods/attributes they have), not their type.

Key idea: Don't check type, check if object has required methods/attributes.
""")


# ----------------------------------------------------------------------------
# 1.2 Duck Typing Example
# ----------------------------------------------------------------------------
print("\n--- 1.2 Duck Typing Example ---")

class Duck:
    """Duck class."""
    
    def quack(self):
        return "Quack!"
    
    def fly(self):
        return "Flying"

class Person:
    """Person class that can quack and fly (duck-like behavior)."""
    
    def quack(self):
        return "I'm quacking like a duck!"
    
    def fly(self):
        return "I'm flying!"

def make_it_quack_and_fly(thing):
    """
    Function that works with any object that has quack() and fly() methods.
    Doesn't check type - uses duck typing.
    """
    print(f"Quack: {thing.quack()}")
    print(f"Fly: {thing.fly()}")

duck = Duck()
person = Person()

print("Duck:")
make_it_quack_and_fly(duck)

print("\nPerson:")
make_it_quack_and_fly(person)

# Both work because they have quack() and fly() methods
# Python doesn't care about the type, only the behavior


# ----------------------------------------------------------------------------
# 1.3 Duck Typing with Different Classes
# ----------------------------------------------------------------------------
print("\n--- 1.3 Duck Typing with Different Classes ---")

class Dog:
    """Dog class."""
    
    def speak(self):
        return "Woof!"
    
    def move(self):
        return "Running"

class Cat:
    """Cat class."""
    
    def speak(self):
        return "Meow!"
    
    def move(self):
        return "Walking"

class Robot:
    """Robot class (not an animal, but has same methods)."""
    
    def speak(self):
        return "Beep boop!"
    
    def move(self):
        return "Rolling"

def interact_with_creature(creature):
    """
    Works with any object that has speak() and move() methods.
    Duck typing in action!
    """
    print(f"Creature speaks: {creature.speak()}")
    print(f"Creature moves: {creature.move()}")

animals = [Dog(), Cat(), Robot()]

print("Different creatures:")
for creature in animals:
    interact_with_creature(creature)
    print()


# ----------------------------------------------------------------------------
# 1.4 Duck Typing vs Type Checking
# ----------------------------------------------------------------------------
print("\n--- 1.4 Duck Typing vs Type Checking ---")

# Duck typing approach (Pythonic)
def process_data_duck_typing(data):
    """Process data using duck typing."""
    # Check if object has required methods, not type
    if hasattr(data, 'read') and hasattr(data, 'write'):
        return "File-like object"
    elif hasattr(data, '__iter__'):
        return "Iterable object"
    else:
        return "Unknown object"

# Type checking approach (less Pythonic)
def process_data_type_checking(data):
    """Process data using type checking."""
    from io import IOBase
    if isinstance(data, IOBase):
        return "File-like object"
    elif isinstance(data, (list, tuple, str)):
        return "Iterable object"
    else:
        return "Unknown object"

# Duck typing is more flexible
class CustomFile:
    """Custom class that behaves like a file."""
    
    def read(self):
        return "Reading..."
    
    def write(self, data):
        return f"Writing: {data}"

custom_file = CustomFile()
print(f"Duck typing: {process_data_duck_typing(custom_file)}")
print(f"Type checking: {process_data_type_checking(custom_file)}")


# ----------------------------------------------------------------------------
# 1.5 Duck Typing in Practice
# ----------------------------------------------------------------------------
print("\n--- 1.5 Duck Typing in Practice ---")

class StringFormatter:
    """String formatter."""
    
    def format(self, text):
        return text.upper()

class NumberFormatter:
    """Number formatter."""
    
    def format(self, number):
        return f"Number: {number}"

class CustomFormatter:
    """Custom formatter."""
    
    def format(self, data):
        return f"Formatted: {data}"

def apply_formatting(formatter, data):
    """
    Works with any object that has format() method.
    Duck typing allows flexibility.
    """
    return formatter.format(data)

formatters = [
    StringFormatter(),
    NumberFormatter(),
    CustomFormatter()
]

data_items = ["hello", 42, "test"]

for formatter, data in zip(formatters, data_items):
    result = apply_formatting(formatter, data)
    print(f"Formatter result: {result}")


# ============================================================================
# 2. METHOD OVERLOADING
# ============================================================================

print("\n" + "=" * 70)
print("2. METHOD OVERLOADING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Python Doesn't Support Method Overloading
# ----------------------------------------------------------------------------
print("\n--- 2.1 Python Doesn't Support Method Overloading ---")
print("""
Python does NOT support method overloading like Java/C++.
You cannot have multiple methods with same name but different signatures.

Last definition wins - previous definitions are overwritten.
""")


# ----------------------------------------------------------------------------
# 2.2 Alternative: Default Parameters
# ----------------------------------------------------------------------------
print("\n--- 2.2 Alternative: Default Parameters ---")

class Calculator:
    """Calculator using default parameters instead of overloading."""
    
    def add(self, a, b=0, c=0):
        """
        Single method handles multiple cases using default parameters.
        Works like: add(a), add(a, b), add(a, b, c)
        """
        return a + b + c

calc = Calculator()
print(f"add(5): {calc.add(5)}")
print(f"add(5, 3): {calc.add(5, 3)}")
print(f"add(5, 3, 2): {calc.add(5, 3, 2)}")


# ----------------------------------------------------------------------------
# 2.3 Alternative: Variable Arguments (*args, **kwargs)
# ----------------------------------------------------------------------------
print("\n--- 2.3 Alternative: Variable Arguments (*args, **kwargs) ---")

class Calculator:
    """Calculator using *args for flexibility."""
    
    def add(self, *args):
        """Add any number of arguments."""
        return sum(args)
    
    def multiply(self, *args):
        """Multiply any number of arguments."""
        result = 1
        for arg in args:
            result *= arg
        return result

calc = Calculator()
print(f"add(1): {calc.add(1)}")
print(f"add(1, 2): {calc.add(1, 2)}")
print(f"add(1, 2, 3, 4, 5): {calc.add(1, 2, 3, 4, 5)}")
print(f"multiply(2, 3, 4): {calc.multiply(2, 3, 4)}")


# ----------------------------------------------------------------------------
# 2.4 Alternative: Type Checking Inside Method
# ----------------------------------------------------------------------------
print("\n--- 2.4 Alternative: Type Checking Inside Method ---")

class Processor:
    """Processor that handles different types."""
    
    def process(self, data):
        """
        Handle different types in one method.
        Mimics method overloading behavior.
        """
        if isinstance(data, str):
            return f"Processing string: {data.upper()}"
        elif isinstance(data, int):
            return f"Processing integer: {data * 2}"
        elif isinstance(data, list):
            return f"Processing list: {len(data)} items"
        else:
            return f"Processing unknown type: {type(data).__name__}"

processor = Processor()
print(f"String: {processor.process('hello')}")
print(f"Integer: {processor.process(42)}")
print(f"List: {processor.process([1, 2, 3])}")


# ----------------------------------------------------------------------------
# 2.5 Alternative: Using functools.singledispatch
# ----------------------------------------------------------------------------
print("\n--- 2.5 Alternative: Using functools.singledispatch ---")

from functools import singledispatch

@singledispatch
def process_data(data):
    """Default implementation."""
    return f"Processing: {data}"

@process_data.register
def _(data: str):
    """Handle strings."""
    return f"Processing string: {data.upper()}"

@process_data.register
def _(data: int):
    """Handle integers."""
    return f"Processing integer: {data * 2}"

@process_data.register
def _(data: list):
    """Handle lists."""
    return f"Processing list: {len(data)} items"

print(f"String: {process_data('hello')}")
print(f"Integer: {process_data(42)}")
print(f"List: {process_data([1, 2, 3])}")
print(f"Default: {process_data(3.14)}")


# ============================================================================
# 3. OPERATOR OVERLOADING
# ============================================================================

print("\n" + "=" * 70)
print("3. OPERATOR OVERLOADING")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Arithmetic Operators
# ----------------------------------------------------------------------------
print("\n--- 3.1 Arithmetic Operators ---")

class Point:
    """Point class with operator overloading."""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        """Overload + operator."""
        if not isinstance(other, Point):
            return NotImplemented
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Overload - operator."""
        if not isinstance(other, Point):
            return NotImplemented
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """Overload * operator (scalar multiplication)."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Point(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar):
        """Right multiplication (scalar * point)."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar):
        """Overload / operator."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Point(self.x / scalar, self.y / scalar)
    
    def __floordiv__(self, scalar):
        """Overload // operator."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Point(self.x // scalar, self.y // scalar)
    
    def __mod__(self, scalar):
        """Overload % operator."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Point(self.x % scalar, self.y % scalar)
    
    def __pow__(self, power):
        """Overload ** operator."""
        if not isinstance(power, (int, float)):
            return NotImplemented
        return Point(self.x ** power, self.y ** power)
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

p1 = Point(3, 4)
p2 = Point(1, 2)

print(f"p1: {p1}")
print(f"p2: {p2}")
print(f"p1 + p2: {p1 + p2}")
print(f"p1 - p2: {p1 - p2}")
print(f"p1 * 2: {p1 * 2}")
print(f"2 * p1: {2 * p1}")  # Uses __rmul__
print(f"p1 / 2: {p1 / 2}")
print(f"p1 ** 2: {p1 ** 2}")


# ----------------------------------------------------------------------------
# 3.2 Comparison Operators
# ----------------------------------------------------------------------------
print("\n--- 3.2 Comparison Operators ---")

class Person:
    """Person class with comparison operators."""
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __eq__(self, other):
        """Overload == operator."""
        if not isinstance(other, Person):
            return NotImplemented
        return self.age == other.age
    
    def __lt__(self, other):
        """Overload < operator."""
        if not isinstance(other, Person):
            return NotImplemented
        return self.age < other.age
    
    def __le__(self, other):
        """Overload <= operator."""
        return self < other or self == other
    
    def __gt__(self, other):
        """Overload > operator."""
        if not isinstance(other, Person):
            return NotImplemented
        return self.age > other.age
    
    def __ge__(self, other):
        """Overload >= operator."""
        return self > other or self == other
    
    def __ne__(self, other):
        """Overload != operator."""
        return not self == other
    
    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

person1 = Person("Alice", 25)
person2 = Person("Bob", 30)
person3 = Person("Charlie", 25)

print(f"person1: {person1}")
print(f"person2: {person2}")
print(f"person3: {person3}")
print(f"\nperson1 == person2: {person1 == person2}")
print(f"person1 == person3: {person1 == person3}")
print(f"person1 < person2: {person1 < person2}")
print(f"person1 <= person2: {person1 <= person2}")
print(f"person1 > person2: {person1 > person2}")

# Can sort objects
people = [person1, person2, person3]
sorted_people = sorted(people)
print(f"\nSorted by age: {sorted_people}")


# ----------------------------------------------------------------------------
# 3.3 Unary Operators
# ----------------------------------------------------------------------------
print("\n--- 3.3 Unary Operators ---")

class Number:
    """Number class with unary operators."""
    
    def __init__(self, value):
        self.value = value
    
    def __neg__(self):
        """Overload - (negation) operator."""
        return Number(-self.value)
    
    def __pos__(self):
        """Overload + (unary plus) operator."""
        return Number(+self.value)
    
    def __abs__(self):
        """Overload abs() function."""
        return Number(abs(self.value))
    
    def __invert__(self):
        """Overload ~ (bitwise NOT) operator."""
        return Number(~int(self.value))
    
    def __repr__(self):
        return f"Number({self.value})"

num = Number(5)
print(f"Number: {num}")
print(f"-num: {-num}")
print(f"+num: {+num}")
print(f"abs(num): {abs(num)}")


# ----------------------------------------------------------------------------
# 3.4 In-Place Operators
# ----------------------------------------------------------------------------
print("\n--- 3.4 In-Place Operators ---")

class Counter:
    """Counter class with in-place operators."""
    
    def __init__(self, value=0):
        self.value = value
    
    def __iadd__(self, other):
        """Overload += operator."""
        self.value += other
        return self
    
    def __isub__(self, other):
        """Overload -= operator."""
        self.value -= other
        return self
    
    def __imul__(self, other):
        """Overload *= operator."""
        self.value *= other
        return self
    
    def __idiv__(self, other):
        """Overload /= operator."""
        self.value /= other
        return self
    
    def __repr__(self):
        return f"Counter({self.value})"

counter = Counter(10)
print(f"Initial: {counter}")

counter += 5
print(f"After += 5: {counter}")

counter -= 3
print(f"After -= 3: {counter}")

counter *= 2
print(f"After *= 2: {counter}")


# ----------------------------------------------------------------------------
# 3.5 Container Operators
# ----------------------------------------------------------------------------
print("\n--- 3.5 Container Operators ---")

class ShoppingList:
    """Shopping list with container operators."""
    
    def __init__(self):
        self.items = []
    
    def __len__(self):
        """Overload len() function."""
        return len(self.items)
    
    def __getitem__(self, index):
        """Overload [] for getting items."""
        return self.items[index]
    
    def __setitem__(self, index, value):
        """Overload [] for setting items."""
        self.items[index] = value
    
    def __delitem__(self, index):
        """Overload del statement."""
        del self.items[index]
    
    def __contains__(self, item):
        """Overload 'in' operator."""
        return item in self.items
    
    def append(self, item):
        """Add item to list."""
        self.items.append(item)
    
    def __repr__(self):
        return f"ShoppingList({self.items})"

shopping = ShoppingList()
shopping.append("Apple")
shopping.append("Banana")
shopping.append("Cherry")

print(f"Shopping list: {shopping}")
print(f"Length: {len(shopping)}")
print(f"First item: {shopping[0]}")
print(f"'Apple' in list: {'Apple' in shopping}")

shopping[1] = "Orange"
print(f"After modification: {shopping}")

del shopping[0]
print(f"After deletion: {shopping}")


# ----------------------------------------------------------------------------
# 3.6 Complete Example: Vector Class
# ----------------------------------------------------------------------------
print("\n--- 3.6 Complete Example: Vector Class ---")

class Vector:
    """Vector class with comprehensive operator overloading."""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        """Vector addition."""
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Vector subtraction."""
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """Scalar multiplication."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar):
        """Right multiplication."""
        return self.__mul__(scalar)
    
    def __abs__(self):
        """Magnitude of vector."""
        import math
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def __eq__(self, other):
        """Vector equality."""
        if not isinstance(other, Vector):
            return NotImplemented
        return self.x == other.x and self.y == other.y
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(f"v1: {v1}")
print(f"v2: {v2}")
print(f"v1 + v2: {v1 + v2}")
print(f"v1 - v2: {v1 - v2}")
print(f"v1 * 2: {v1 * 2}")
print(f"|v1|: {abs(v1)}")
print(f"v1 == v2: {v1 == v2}")


# ============================================================================
# 4. PRACTICAL EXAMPLES
# ============================================================================

print("\n" + "=" * 70)
print("4. PRACTICAL EXAMPLES")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Duck Typing: File-like Objects
# ----------------------------------------------------------------------------
print("\n--- 4.1 Duck Typing: File-like Objects ---")

def read_data(source):
    """
    Works with any object that has read() method.
    Duck typing allows flexibility.
    """
    return source.read()

class StringReader:
    """String reader that behaves like a file."""
    
    def __init__(self, data):
        self.data = data
        self.position = 0
    
    def read(self):
        result = self.data[self.position:]
        self.position = len(self.data)
        return result

class FileReader:
    """File reader."""
    
    def __init__(self, filename):
        self.filename = filename
    
    def read(self):
        return f"Reading from {self.filename}"

# Both work with read_data() function
string_reader = StringReader("Hello World")
file_reader = FileReader("data.txt")

print(f"String reader: {read_data(string_reader)}")
print(f"File reader: {read_data(file_reader)}")


# ----------------------------------------------------------------------------
# 4.2 Operator Overloading: Complex Number
# ----------------------------------------------------------------------------
print("\n--- 4.2 Operator Overloading: Complex Number ---")

class ComplexNumber:
    """Complex number with operator overloading."""
    
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag
    
    def __add__(self, other):
        if not isinstance(other, ComplexNumber):
            return NotImplemented
        return ComplexNumber(self.real + other.real, self.imag + other.imag)
    
    def __mul__(self, other):
        if not isinstance(other, ComplexNumber):
            return NotImplemented
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag + self.imag * other.real
        return ComplexNumber(real, imag)
    
    def __repr__(self):
        return f"{self.real} + {self.imag}i"

c1 = ComplexNumber(1, 2)
c2 = ComplexNumber(3, 4)

print(f"c1: {c1}")
print(f"c2: {c2}")
print(f"c1 + c2: {c1 + c2}")
print(f"c1 * c2: {c1 * c2}")


# ============================================================================
# 5. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("5. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. DUCK TYPING:
   - "If it walks like a duck and quacks like a duck, it's a duck"
   - Python focuses on behavior, not type
   - Check for methods/attributes, not type
   - More flexible and Pythonic than type checking
   - Allows polymorphism without inheritance

2. METHOD OVERLOADING:
   - Python does NOT support method overloading
   - Last definition overwrites previous ones
   - Alternatives:
     * Default parameters
     * *args and **kwargs
     * Type checking inside method
     * functools.singledispatch

3. OPERATOR OVERLOADING:
   - Implement magic methods to overload operators
   - Arithmetic: __add__, __sub__, __mul__, __truediv__, etc.
   - Comparison: __eq__, __lt__, __gt__, __le__, __ge__, __ne__
   - Unary: __neg__, __pos__, __abs__, __invert__
   - In-place: __iadd__, __isub__, __imul__, etc.
   - Container: __len__, __getitem__, __setitem__, __contains__

4. COMMON OPERATOR METHODS:
   - __add__(self, other): +
   - __sub__(self, other): -
   - __mul__(self, other): *
   - __truediv__(self, other): /
   - __eq__(self, other): ==
   - __lt__(self, other): <
   - __len__(self): len()
   - __getitem__(self, key): []
   - __contains__(self, item): in

5. RETURNING NotImplemented:
   - Return NotImplemented when operation not supported
   - Python will try reverse operation (__radd__, etc.)
   - Allows for flexible operator overloading

6. DUCK TYPING BENEFITS:
   - More flexible code
   - Works with any object that has required methods
   - No need for common base class
   - Easier to extend functionality

7. BEST PRACTICES:
   - Use duck typing when possible
   - Implement operator overloading when it makes sense
   - Return NotImplemented for unsupported operations
   - Keep operator behavior intuitive
   - Document operator behavior

8. INTERVIEW TIPS:
   - Understand duck typing concept
   - Know Python doesn't support method overloading
   - Know alternatives to method overloading
   - Be able to implement common operator overloads
   - Understand when to use duck typing vs type checking
   - Know how NotImplemented works
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Polymorphism Guide Ready!")
    print("=" * 70)
