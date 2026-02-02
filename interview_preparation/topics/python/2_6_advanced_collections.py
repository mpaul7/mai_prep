"""
Python Advanced Collections - Interview Preparation
Topic 2.6: Advanced Collections

This module covers:
- collections.Counter: Frequency counting
- collections.defaultdict: Default values
- collections.deque: Double-ended queue
- collections.namedtuple: Named tuples
"""

from collections import Counter, defaultdict, deque, namedtuple

# ============================================================================
# 1. COLLECTIONS.COUNTER
# ============================================================================

print("=" * 70)
print("1. COLLECTIONS.COUNTER")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 Creating Counter
# ----------------------------------------------------------------------------
print("\n--- Creating Counter ---")

# From iterable
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
counter = Counter(words)
print(f"Counter from list: {counter}")

# From string
text = "hello"
char_counter = Counter(text)
print(f"Counter from string: {char_counter}")

# From dictionary
counter = Counter({"a": 3, "b": 2, "c": 1})
print(f"Counter from dict: {counter}")

# Empty counter
empty_counter = Counter()
print(f"Empty counter: {empty_counter}")


# ----------------------------------------------------------------------------
# 1.2 Accessing Counter Elements
# ----------------------------------------------------------------------------
print("\n--- Accessing Counter Elements ---")

counter = Counter(["apple", "banana", "apple", "cherry"])

# Access count of element
print(f"Count of 'apple': {counter['apple']}")
print(f"Count of 'banana': {counter['banana']}")

# Missing elements return 0 (no KeyError)
print(f"Count of 'orange': {counter['orange']}")  # 0

# Counter is a dictionary subclass
print(f"Is dict: {isinstance(counter, dict)}")
print(f"Keys: {list(counter.keys())}")
print(f"Values: {list(counter.values())}")
print(f"Items: {list(counter.items())}")


# ----------------------------------------------------------------------------
# 1.3 Updating Counter
# ----------------------------------------------------------------------------
print("\n--- Updating Counter ---")

counter = Counter(["apple", "banana"])

# Update with iterable
counter.update(["apple", "cherry", "apple"])
print(f"After update: {counter}")

# Update with another Counter
counter.update(Counter({"banana": 2, "date": 1}))
print(f"After Counter update: {counter}")

# Update with dictionary
counter.update({"apple": 1, "elderberry": 1})
print(f"After dict update: {counter}")

# Direct assignment (modify count)
counter["apple"] = 5
print(f"After direct assignment: {counter}")


# ----------------------------------------------------------------------------
# 1.4 Counter Methods
# ----------------------------------------------------------------------------
print("\n--- Counter Methods ---")

counter = Counter(["apple", "banana", "apple", "cherry", "banana", "apple"])
print(f"Counter: {counter}")

# most_common() - Get most common elements
print(f"Most common 2: {counter.most_common(2)}")
print(f"Most common (all): {counter.most_common()}")

# elements() - Get all elements (as iterator)
elements = list(counter.elements())
print(f"Elements: {elements}")

# subtract() - Subtract counts
counter.subtract(["apple", "banana"])
print(f"After subtract: {counter}")

# total() - Total count (Python 3.10+)
# total = counter.total()
# print(f"Total: {total}")


# ----------------------------------------------------------------------------
# 1.5 Counter Operations
# ----------------------------------------------------------------------------
print("\n--- Counter Operations ---")

counter1 = Counter(["a", "b", "c", "a"])
counter2 = Counter(["a", "b", "b"])

print(f"Counter1: {counter1}")
print(f"Counter2: {counter2}")

# Addition (combine counts)
combined = counter1 + counter2
print(f"Counter1 + Counter2: {combined}")

# Subtraction (only positive counts)
subtracted = counter1 - counter2
print(f"Counter1 - Counter2: {subtracted}")

# Intersection (minimum counts)
intersection = counter1 & counter2
print(f"Counter1 & Counter2: {intersection}")

# Union (maximum counts)
union = counter1 | counter2
print(f"Counter1 | Counter2: {union}")


# ----------------------------------------------------------------------------
# 1.6 Common Counter Use Cases
# ----------------------------------------------------------------------------
print("\n--- Common Counter Use Cases ---")

# Count character frequencies
text = "hello world"
char_count = Counter(text)
print(f"Character frequencies: {char_count}")

# Count word frequencies
words = "the quick brown fox jumps over the lazy dog".split()
word_count = Counter(words)
print(f"Word frequencies: {word_count}")

# Find most common elements
numbers = [1, 2, 3, 2, 3, 3, 4, 4, 4, 4]
num_count = Counter(numbers)
print(f"Most common 2 numbers: {num_count.most_common(2)}")

# Compare two sequences
list1 = [1, 2, 3, 2, 3]
list2 = [2, 3, 3, 4]
counter1 = Counter(list1)
counter2 = Counter(list2)
print(f"Common elements: {counter1 & counter2}")


# ============================================================================
# 2. COLLECTIONS.DEFAULTDICT
# ============================================================================

print("\n" + "=" * 70)
print("2. COLLECTIONS.DEFAULTDICT")
print("=" * 70)

# Note: defaultdict was covered in detail in 2_3_dictionaries.py
# This section provides a quick reference

# ----------------------------------------------------------------------------
# 2.1 Creating defaultdict
# ----------------------------------------------------------------------------
print("\n--- Creating defaultdict ---")

# Default to int (0)
int_dict = defaultdict(int)
print(f"defaultdict(int)['missing']: {int_dict['missing']}")

# Default to list ([])
list_dict = defaultdict(list)
list_dict["fruits"].append("apple")
print(f"defaultdict(list): {list_dict}")

# Default to dict ({})
dict_dict = defaultdict(dict)
dict_dict["person"]["name"] = "Alice"
print(f"defaultdict(dict): {dict_dict}")

# Default to set (set())
set_dict = defaultdict(set)
set_dict["tags"].add("python")
print(f"defaultdict(set): {set_dict}")


# ----------------------------------------------------------------------------
# 2.2 Common defaultdict Patterns
# ----------------------------------------------------------------------------
print("\n--- Common defaultdict Patterns ---")

# Grouping items
data = [("fruit", "apple"), ("fruit", "banana"), ("vegetable", "carrot")]
groups = defaultdict(list)
for category, item in data:
    groups[category].append(item)
print(f"Grouped: {groups}")

# Counting
items = ["a", "b", "a", "c", "b", "a"]
counts = defaultdict(int)
for item in items:
    counts[item] += 1
print(f"Counts: {counts}")

# Nested structures
nested = defaultdict(dict)
nested["user1"]["name"] = "Alice"
nested["user1"]["age"] = 25
print(f"Nested: {nested}")


# ----------------------------------------------------------------------------
# 2.3 defaultdict vs Regular Dict
# ----------------------------------------------------------------------------
print("\n--- defaultdict vs Regular Dict ---")

# Regular dict approach
def count_regular(items):
    """Count using regular dict."""
    counts = {}
    for item in items:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    return counts

# defaultdict approach
def count_defaultdict(items):
    """Count using defaultdict."""
    counts = defaultdict(int)
    for item in items:
        counts[item] += 1  # No need to check
    return counts

items = ["a", "b", "a", "c"]
print(f"Regular dict: {count_regular(items)}")
print(f"defaultdict: {count_defaultdict(items)}")


# ============================================================================
# 3. COLLECTIONS.DEQUE
# ============================================================================

print("\n" + "=" * 70)
print("3. COLLECTIONS.DEQUE")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Creating deque
# ----------------------------------------------------------------------------
print("\n--- Creating deque ---")

# Empty deque
d = deque()
print(f"Empty deque: {d}")

# From iterable
d = deque([1, 2, 3, 4, 5])
print(f"Deque from list: {d}")

# With maxlen (bounded deque)
bounded = deque([1, 2, 3], maxlen=5)
print(f"Bounded deque: {bounded}")


# ----------------------------------------------------------------------------
# 3.2 Adding Elements
# ----------------------------------------------------------------------------
print("\n--- Adding Elements ---")

d = deque([1, 2, 3])

# Append to right (end)
d.append(4)
print(f"After append(4): {d}")

# Append to left (beginning)
d.appendleft(0)
print(f"After appendleft(0): {d}")

# Extend right
d.extend([5, 6])
print(f"After extend([5, 6]): {d}")

# Extend left (note: extends in reverse order)
d.extendleft([-1, -2])
print(f"After extendleft([-1, -2]): {d}")


# ----------------------------------------------------------------------------
# 3.3 Removing Elements
# ----------------------------------------------------------------------------
print("\n--- Removing Elements ---")

d = deque([1, 2, 3, 4, 5])
print(f"Original: {d}")

# Pop from right (end)
right = d.pop()
print(f"After pop(): {d}, popped: {right}")

# Pop from left (beginning)
left = d.popleft()
print(f"After popleft(): {d}, popped: {left}")

# Remove specific element
d = deque([1, 2, 3, 2, 4])
d.remove(2)  # Removes first occurrence
print(f"After remove(2): {d}")

# Clear all elements
d.clear()
print(f"After clear(): {d}")


# ----------------------------------------------------------------------------
# 3.4 Accessing Elements
# ----------------------------------------------------------------------------
print("\n--- Accessing Elements ---")

d = deque([10, 20, 30, 40, 50])

# Indexing (like list)
print(f"d[0] = {d[0]}")
print(f"d[-1] = {d[-1]}")
print(f"d[2] = {d[2]}")

# Slicing (deques don't support slicing directly - convert to list first)
print(f"d[1:4] = {deque(list(d)[1:4])}")

# Length
print(f"Length: {len(d)}")

# Count occurrences
d = deque([1, 2, 2, 3, 2])
print(f"Count of 2: {d.count(2)}")


# ----------------------------------------------------------------------------
# 3.5 Rotating deque
# ----------------------------------------------------------------------------
print("\n--- Rotating deque ---")

d = deque([1, 2, 3, 4, 5])
print(f"Original: {d}")

# Rotate right (positive)
d.rotate(2)
print(f"After rotate(2): {d}")

# Rotate left (negative)
d.rotate(-1)
print(f"After rotate(-1): {d}")


# ----------------------------------------------------------------------------
# 3.6 Bounded deque (maxlen)
# ----------------------------------------------------------------------------
print("\n--- Bounded deque (maxlen) ---")

# Create bounded deque
d = deque([1, 2, 3], maxlen=3)
print(f"Bounded deque (maxlen=3): {d}")

# Adding to full deque removes from other end
d.append(4)
print(f"After append(4): {d}")  # [2, 3, 4] - 1 was removed

d.appendleft(0)
print(f"After appendleft(0): {d}")  # [0, 2, 3] - 4 was removed


# ----------------------------------------------------------------------------
# 3.7 deque vs list
# ----------------------------------------------------------------------------
print("\n--- deque vs list ---")

# deque is optimized for append/pop from both ends
# O(1) for append/pop from both ends
# O(n) for operations in middle

# list is optimized for operations at end
# O(1) for append/pop at end
# O(n) for operations at beginning or middle

# Use deque for:
# - Queues (FIFO)
# - Stacks (LIFO)
# - When you need fast operations at both ends

# Use list for:
# - Random access
# - Operations mostly at end
# - When you need slicing


# ----------------------------------------------------------------------------
# 3.8 Common deque Use Cases
# ----------------------------------------------------------------------------
print("\n--- Common deque Use Cases ---")

# Queue (FIFO)
queue = deque()
queue.append(1)  # Enqueue
queue.append(2)
queue.append(3)
print(f"Queue: {queue}")
first = queue.popleft()  # Dequeue
print(f"Dequeued: {first}, Remaining: {queue}")

# Stack (LIFO)
stack = deque()
stack.append(1)  # Push
stack.append(2)
stack.append(3)
print(f"Stack: {stack}")
top = stack.pop()  # Pop
print(f"Popped: {top}, Remaining: {stack}")

# Sliding window
def sliding_window_max(nums, k):
    """Find max in each sliding window."""
    dq = deque()
    result = []
    for i, num in enumerate(nums):
        # Remove indices outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
        # Remove smaller elements
        while dq and nums[dq[-1]] <= num:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result

nums = [1, 3, -1, -3, 5, 3, 6, 7]
print(f"Sliding window max (k=3): {sliding_window_max(nums, 3)}")


# ============================================================================
# 4. COLLECTIONS.NAMEDTUPLE
# ============================================================================

print("\n" + "=" * 70)
print("4. COLLECTIONS.NAMEDTUPLE")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 Creating namedtuple
# ----------------------------------------------------------------------------
print("\n--- Creating namedtuple ---")

# Define namedtuple type
Point = namedtuple("Point", ["x", "y"])

# Create instance
p1 = Point(1, 2)
print(f"Point(1, 2): {p1}")
print(f"p1.x = {p1.x}, p1.y = {p1.y}")

# Access by index (like regular tuple)
print(f"p1[0] = {p1[0]}, p1[1] = {p1[1]}")

# Unpacking
x, y = p1
print(f"Unpacked: x={x}, y={y}")


# ----------------------------------------------------------------------------
# 4.2 namedtuple Features
# ----------------------------------------------------------------------------
print("\n--- namedtuple Features ---")

# Define Person namedtuple
Person = namedtuple("Person", ["name", "age", "city"])

# Create instances
person1 = Person("Alice", 25, "New York")
person2 = Person("Bob", 30, "London")

print(f"Person1: {person1}")
print(f"Person1.name = {person1.name}")
print(f"Person1.age = {person1.age}")

# Immutable (like tuple)
# person1.age = 26  # AttributeError: can't set attribute

# _asdict() - Convert to dictionary
person_dict = person1._asdict()
print(f"As dict: {person_dict}")

# _replace() - Create new instance with changed values
person1_updated = person1._replace(age=26)
print(f"Updated: {person1_updated}")


# ----------------------------------------------------------------------------
# 4.3 namedtuple with Default Values
# ----------------------------------------------------------------------------
print("\n--- namedtuple with Default Values ---")

# Define with defaults (Python 3.7+)
from collections import namedtuple

# Method 1: Using defaults parameter
Person = namedtuple("Person", ["name", "age", "city"], defaults=["Unknown", "Unknown"])
person1 = Person("Alice")
print(f"Person with defaults: {person1}")

# Method 2: Using _field_defaults
Person = namedtuple("Person", ["name", "age", "city"])
Person.__new__.__defaults__ = (None, None)
person2 = Person("Bob")
print(f"Person with field defaults: {person2}")


# ----------------------------------------------------------------------------
# 4.4 namedtuple Methods
# ----------------------------------------------------------------------------
print("\n--- namedtuple Methods ---")

Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4)

# _fields - Get field names
print(f"Fields: {p._fields}")

# _asdict() - Convert to OrderedDict
point_dict = p._asdict()
print(f"As dict: {point_dict}")

# _replace() - Create new instance
p_new = p._replace(x=5)
print(f"Original: {p}, New: {p_new}")

# _make() - Create from iterable
p_from_iter = Point._make([10, 20])
print(f"From iterable: {p_from_iter}")


# ----------------------------------------------------------------------------
# 4.5 namedtuple Use Cases
# ----------------------------------------------------------------------------
print("\n--- namedtuple Use Cases ---")

# Representing data records
Employee = namedtuple("Employee", ["name", "id", "department", "salary"])
emp1 = Employee("Alice", 101, "Engineering", 100000)
emp2 = Employee("Bob", 102, "Sales", 80000)

print(f"Employee 1: {emp1}")
print(f"Employee 1 salary: {emp1.salary}")

# Coordinates
Point3D = namedtuple("Point3D", ["x", "y", "z"])
p1 = Point3D(1, 2, 3)
p2 = Point3D(4, 5, 6)

# Calculate distance
def distance(p1, p2):
    """Calculate distance between two 3D points."""
    return ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2 + (p2.z - p1.z) ** 2) ** 0.5

dist = distance(p1, p2)
print(f"Distance between {p1} and {p2}: {dist:.2f}")

# Configuration settings
Config = namedtuple("Config", ["host", "port", "debug"])
config = Config("localhost", 8080, True)
print(f"Config: {config}")


# ----------------------------------------------------------------------------
# 4.6 namedtuple vs Class vs Dictionary
# ----------------------------------------------------------------------------
print("\n--- namedtuple vs Class vs Dictionary ---")

# namedtuple
Person_nt = namedtuple("Person", ["name", "age"])
p_nt = Person_nt("Alice", 25)

# Dictionary
p_dict = {"name": "Alice", "age": 25}

# Class
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

p_class = Person("Alice", 25)

print(f"namedtuple: {p_nt.name}, {p_nt.age}")
print(f"Dictionary: {p_dict['name']}, {p_dict['age']}")
print(f"Class: {p_class.name}, {p_class.age}")

# namedtuple advantages:
# - Immutable (data integrity)
# - Memory efficient
# - Can use as dictionary keys
# - Clean syntax
# - Tuple-like behavior

# Use namedtuple when:
# - Need immutable data structure
# - Want tuple-like behavior with named fields
# - Need to use as dictionary keys
# - Want memory efficiency


# ============================================================================
# 5. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: Count word frequencies
print("\n--- Exercise 1: Count Word Frequencies ---")
def count_words(text):
    """Count word frequencies using Counter."""
    words = text.lower().split()
    return Counter(words)

text = "the quick brown fox jumps over the lazy dog the quick"
word_count = count_words(text)
print(f"Word frequencies: {word_count}")
print(f"Most common 3: {word_count.most_common(3)}")


# Exercise 2: Find most frequent character
print("\n--- Exercise 2: Most Frequent Character ---")
def most_frequent_char(text):
    """Find most frequent character."""
    char_count = Counter(text.replace(" ", ""))
    return char_count.most_common(1)[0]

text = "hello world"
char, count = most_frequent_char(text)
print(f"Most frequent char in '{text}': '{char}' ({count} times)")


# Exercise 3: Group items by category
print("\n--- Exercise 3: Group Items ---")
def group_items(items):
    """Group items by category using defaultdict."""
    groups = defaultdict(list)
    for category, item in items:
        groups[category].append(item)
    return dict(groups)

items = [("fruit", "apple"), ("fruit", "banana"), ("vegetable", "carrot")]
grouped = group_items(items)
print(f"Grouped items: {grouped}")


# Exercise 4: Implement queue with deque
print("\n--- Exercise 4: Queue Implementation ---")
class Queue:
    """Queue implementation using deque."""
    def __init__(self):
        self._items = deque()
    
    def enqueue(self, item):
        """Add item to queue."""
        self._items.append(item)
    
    def dequeue(self):
        """Remove and return item from queue."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items.popleft()
    
    def is_empty(self):
        """Check if queue is empty."""
        return len(self._items) == 0
    
    def size(self):
        """Get queue size."""
        return len(self._items)

queue = Queue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(f"Queue size: {queue.size()}")
print(f"Dequeued: {queue.dequeue()}")
print(f"Queue size after dequeue: {queue.size()}")


# Exercise 5: Create Point namedtuple
print("\n--- Exercise 5: Point namedtuple ---")
Point = namedtuple("Point", ["x", "y"])

def midpoint(p1, p2):
    """Calculate midpoint between two points."""
    return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)

p1 = Point(0, 0)
p2 = Point(4, 4)
mid = midpoint(p1, p2)
print(f"Midpoint of {p1} and {p2}: {mid}")


# Exercise 6: Rotate deque
print("\n--- Exercise 6: Rotate Deque ---")
def rotate_array(arr, k):
    """Rotate array k positions to the right using deque."""
    d = deque(arr)
    d.rotate(k)
    return list(d)

arr = [1, 2, 3, 4, 5]
rotated = rotate_array(arr, 2)
print(f"Rotated {arr} by 2: {rotated}")


# Exercise 7: Counter operations
print("\n--- Exercise 7: Counter Operations ---")
def find_common_elements(list1, list2):
    """Find common elements using Counter."""
    counter1 = Counter(list1)
    counter2 = Counter(list2)
    return list((counter1 & counter2).elements())

list1 = [1, 2, 3, 2, 3]
list2 = [2, 3, 3, 4]
common = find_common_elements(list1, list2)
print(f"Common elements: {common}")


# Exercise 8: Student record with namedtuple
print("\n--- Exercise 8: Student Record ---")
Student = namedtuple("Student", ["name", "id", "grades"])

def calculate_average(student):
    """Calculate average grade."""
    return sum(student.grades) / len(student.grades) if student.grades else 0

student = Student("Alice", 101, [85, 90, 88, 92])
avg = calculate_average(student)
print(f"Student: {student.name}, Average: {avg:.2f}")


# ============================================================================
# 6. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("6. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# Q1: When to use Counter?
print("\n--- Q1: When to Use Counter? ---")
print("""
Use Counter when:
- Counting frequencies of elements
- Finding most common elements
- Comparing frequencies between sequences
- Need dictionary-like interface with counting operations
Counter is optimized for counting operations
""")


# Q2: What's the difference between Counter and defaultdict(int)?
print("\n--- Q2: Counter vs defaultdict(int) ---")
print("""
Counter:
- Specialized for counting
- Has methods like most_common(), elements()
- Supports arithmetic operations (+, -, &, |)
- Automatically handles missing keys (returns 0)

defaultdict(int):
- General purpose default dictionary
- Need to manually implement counting logic
- No special counting methods
- Also handles missing keys (returns 0)

Use Counter for counting, defaultdict(int) for general defaults
""")


# Q3: When to use deque vs list?
print("\n--- Q3: deque vs list ---")
print("""
deque:
- Fast append/pop from both ends O(1)
- Use for queues, stacks
- Use when operations at both ends needed
- Slower for operations in middle

list:
- Fast append/pop at end O(1)
- Fast random access O(1)
- Good for operations mostly at end
- Slower for operations at beginning O(n)

Use deque for queue/stack operations, list for general purpose
""")


# Q4: What are namedtuples?
print("\n--- Q4: namedtuples ---")
print("""
namedtuple: Tuple subclass with named fields
- Immutable like tuples
- Access by name or index
- Memory efficient
- Can use as dictionary keys
- Cleaner than regular tuples for structured data
Use when you need immutable records with named fields
""")


# Q5: How to update a Counter?
print("\n--- Q5: Updating Counter ---")
print("""
Methods to update:
1. update(iterable) - Add counts from iterable
2. subtract(iterable) - Subtract counts
3. Direct assignment - counter[key] = value
4. Arithmetic operations - counter1 + counter2
update() is most common for adding counts
""")


# Q6: Can deque be used as a stack?
print("\n--- Q6: deque as Stack ---")
print("""
Yes, deque can be used as stack:
- append() for push
- pop() for pop
- Both O(1) operations
- More efficient than list for stack operations
- Can also use as queue with append/popleft
""")


# Q7: Are namedtuples mutable?
print("\n--- Q7: namedtuple Mutability ---")
print("""
No, namedtuples are immutable like regular tuples
Cannot modify fields after creation
Use _replace() to create new instance with changed values
This ensures data integrity
""")


# ============================================================================
# 7. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("7. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. COLLECTIONS.COUNTER:
   - Specialized for frequency counting
   - Methods: most_common(), elements(), update(), subtract()
   - Supports arithmetic: +, -, &, |
   - Missing keys return 0 (no KeyError)
   - Use for counting, finding most common elements

2. COLLECTIONS.DEFAULTDICT:
   - Provides default values for missing keys
   - Common factories: int, list, dict, set
   - Eliminates need for key existence checks
   - Use for grouping, counting, nested structures

3. COLLECTIONS.DEQUE:
   - Double-ended queue
   - O(1) append/pop from both ends
   - Methods: append, appendleft, pop, popleft, rotate
   - Can be bounded with maxlen
   - Use for queues, stacks, sliding windows

4. COLLECTIONS.NAMEDTUPLE:
   - Tuple with named fields
   - Immutable, memory efficient
   - Access by name or index
   - Methods: _asdict(), _replace(), _fields
   - Use for structured data, records, coordinates

5. WHEN TO USE EACH:
   - Counter: Frequency counting, most common elements
   - defaultdict: Grouping, counting, nested structures
   - deque: Queues, stacks, operations at both ends
   - namedtuple: Immutable records, structured data

6. PERFORMANCE:
   - Counter: Optimized for counting operations
   - defaultdict: Same as dict, but with defaults
   - deque: O(1) at both ends, O(n) in middle
   - namedtuple: Memory efficient, fast access

7. BEST PRACTICES:
   - Use Counter for counting (not manual dict)
   - Use defaultdict to avoid key checks
   - Use deque for queue/stack operations
   - Use namedtuple for immutable structured data
   - Know when each collection is appropriate

8. COMMON PATTERNS:
   - Counter: word_count = Counter(words)
   - defaultdict: groups = defaultdict(list)
   - deque: queue = deque(); queue.append/popleft
   - namedtuple: Point = namedtuple("Point", ["x", "y"])
""")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("All examples executed successfully!")
    print("=" * 70)
