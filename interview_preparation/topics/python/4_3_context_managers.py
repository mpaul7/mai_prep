"""
Python Context Managers - Interview Preparation
Topic 4.3: Context Managers

This module covers:
- with Statement: File handling, resource management
- Custom Context Managers: __enter__, __exit__
- contextlib: @contextmanager decorator
"""

import contextlib
import time
import threading
from contextlib import contextmanager, redirect_stdout, redirect_stderr, suppress

# ============================================================================
# 1. UNDERSTANDING CONTEXT MANAGERS
# ============================================================================

print("=" * 70)
print("1. UNDERSTANDING CONTEXT MANAGERS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1.1 What are Context Managers?
# ----------------------------------------------------------------------------
print("\n--- What are Context Managers? ---")

print("""
Context managers are objects that define what happens when you enter and exit
a 'with' statement. They ensure proper setup and cleanup of resources.

Key benefits:
- Automatic resource cleanup (files, locks, connections)
- Exception-safe resource management
- Cleaner code than try-finally blocks
- Ensures __exit__ is always called, even on exceptions
""")


# ----------------------------------------------------------------------------
# 1.2 The Problem They Solve
# ----------------------------------------------------------------------------
print("\n--- The Problem They Solve ---")

# Without context manager (manual cleanup)
print("Without context manager:")
file = open('example.txt', 'w')
try:
    file.write("Hello, World!")
finally:
    file.close()  # Must remember to close!

# With context manager (automatic cleanup)
print("\nWith context manager:")
with open('example.txt', 'w') as file:
    file.write("Hello, World!")
# File automatically closed when exiting 'with' block


# ============================================================================
# 2. THE WITH STATEMENT
# ============================================================================

print("\n" + "=" * 70)
print("2. THE WITH STATEMENT")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 Basic with Statement
# ----------------------------------------------------------------------------
print("\n--- Basic with Statement ---")

# File handling (most common use case)
with open('example.txt', 'w') as f:
    f.write("Hello, World!")
    print("File written successfully")
# File is automatically closed here

# Reading from file
with open('example.txt', 'r') as f:
    content = f.read()
    print(f"File content: {content}")
# File is automatically closed here


# ----------------------------------------------------------------------------
# 2.2 Multiple Context Managers
# ----------------------------------------------------------------------------
print("\n--- Multiple Context Managers ---")

# Reading from one file and writing to another
with open('example.txt', 'r') as input_file, open('output.txt', 'w') as output_file:
    content = input_file.read()
    output_file.write(content.upper())
    print("Files processed")
# Both files automatically closed


# ----------------------------------------------------------------------------
# 2.3 Nested with Statements
# ----------------------------------------------------------------------------
print("\n--- Nested with Statements ---")

with open('example.txt', 'r') as f1:
    with open('output.txt', 'w') as f2:
        content = f1.read()
        f2.write(content)
        print("Nested context managers")
# Both files closed when exiting outer block


# ----------------------------------------------------------------------------
# 2.4 with Statement and Exceptions
# ----------------------------------------------------------------------------
print("\n--- with Statement and Exceptions ---")

# Context manager ensures cleanup even if exception occurs
try:
    with open('example.txt', 'r') as f:
        content = f.read()
        # Simulate an error
        # raise ValueError("Something went wrong")
        print("File read successfully")
except ValueError as e:
    print(f"Exception occurred: {e}")
# File is still closed even if exception occurs!


# ============================================================================
# 3. CUSTOM CONTEXT MANAGERS (Class-based)
# ============================================================================

print("\n" + "=" * 70)
print("3. CUSTOM CONTEXT MANAGERS (Class-based)")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 Basic Custom Context Manager
# ----------------------------------------------------------------------------
print("\n--- Basic Custom Context Manager ---")

class FileManager:
    """Custom context manager for file operations."""
    
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        """Called when entering 'with' block."""
        print(f"Opening file: {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file  # Return value assigned to variable after 'as'
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Called when exiting 'with' block."""
        if self.file:
            print(f"Closing file: {self.filename}")
            self.file.close()
        
        # Return False to propagate exceptions, True to suppress them
        return False

# Using custom context manager
with FileManager('example.txt', 'w') as f:
    f.write("Hello from custom context manager!")
    print("File written")


# ----------------------------------------------------------------------------
# 3.2 Understanding __exit__ Parameters
# ----------------------------------------------------------------------------
print("\n--- Understanding __exit__ Parameters ---")

class VerboseContextManager:
    """Context manager that shows exception handling."""
    
    def __enter__(self):
        print("Entering context")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Exiting context")
        print(f"Exception type: {exc_type}")
        print(f"Exception value: {exc_value}")
        
        if exc_type is not None:
            print("Exception occurred!")
            # Return False to propagate exception
            # Return True to suppress exception
            return False
        
        return False

with VerboseContextManager() as cm:
    print("Inside context")
    # Uncomment to see exception handling:
    # raise ValueError("Test exception")


# ----------------------------------------------------------------------------
# 3.3 Context Manager with Exception Suppression
# ----------------------------------------------------------------------------
print("\n--- Context Manager with Exception Suppression ---")

class SuppressExceptions:
    """Context manager that suppresses exceptions."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        # Return True to suppress all exceptions
        return True

with SuppressExceptions():
    print("This will execute")
    # raise ValueError("This exception will be suppressed")
    print("This will also execute")


# ----------------------------------------------------------------------------
# 3.4 Timer Context Manager
# ----------------------------------------------------------------------------
print("\n--- Timer Context Manager ---")

class Timer:
    """Context manager that measures execution time."""
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        print(f"Elapsed time: {self.elapsed:.4f} seconds")
        return False

with Timer():
    time.sleep(0.1)
    print("Doing some work...")


# ----------------------------------------------------------------------------
# 3.5 Lock Context Manager
# ----------------------------------------------------------------------------
print("\n--- Lock Context Manager ---")

class ThreadLock:
    """Context manager for thread synchronization."""
    
    def __init__(self):
        self.lock = threading.Lock()
    
    def __enter__(self):
        self.lock.acquire()
        print("Lock acquired")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.lock.release()
        print("Lock released")
        return False

# Using lock context manager
lock = ThreadLock()
with lock:
    print("Critical section - only one thread can be here")
    time.sleep(0.01)


# ----------------------------------------------------------------------------
# 3.6 Database Connection Context Manager
# ----------------------------------------------------------------------------
print("\n--- Database Connection Context Manager ---")

class DatabaseConnection:
    """Simulated database connection context manager."""
    
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connected = False
    
    def __enter__(self):
        print(f"Connecting to database at {self.host}:{self.port}")
        self.connected = True
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.connected:
            print("Closing database connection")
            self.connected = False
        return False
    
    def query(self, sql):
        """Execute a query."""
        if not self.connected:
            raise RuntimeError("Not connected to database")
        print(f"Executing: {sql}")
        return "Query results"

with DatabaseConnection("localhost", 5432) as db:
    results = db.query("SELECT * FROM users")
    print(f"Results: {results}")


# ----------------------------------------------------------------------------
# 3.7 Change Directory Context Manager
# ----------------------------------------------------------------------------
print("\n--- Change Directory Context Manager ---")

import os

class ChangeDirectory:
    """Context manager that temporarily changes directory."""
    
    def __init__(self, new_dir):
        self.new_dir = new_dir
        self.old_dir = None
    
    def __enter__(self):
        self.old_dir = os.getcwd()
        os.chdir(self.new_dir)
        print(f"Changed directory to: {os.getcwd()}")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.old_dir)
        print(f"Restored directory to: {os.getcwd()}")
        return False

# Note: This is a demonstration - actual directory changes depend on filesystem
print("Current directory:", os.getcwd())


# ============================================================================
# 4. CONTEXTLIB MODULE
# ============================================================================

print("\n" + "=" * 70)
print("4. CONTEXTLIB MODULE")
print("=" * 70)

# ----------------------------------------------------------------------------
# 4.1 @contextmanager Decorator
# ----------------------------------------------------------------------------
print("\n--- @contextmanager Decorator ---")

# Using @contextmanager to create context manager from generator function
@contextmanager
def file_manager(filename, mode='r'):
    """Context manager using @contextmanager decorator."""
    print(f"Opening file: {filename}")
    file = open(filename, mode)
    try:
        yield file  # Value returned to 'as' variable
    finally:
        print(f"Closing file: {filename}")
        file.close()

with file_manager('example.txt', 'w') as f:
    f.write("Hello from @contextmanager!")
    print("File written")


# ----------------------------------------------------------------------------
# 4.2 Timer with @contextmanager
# ----------------------------------------------------------------------------
print("\n--- Timer with @contextmanager ---")

@contextmanager
def timer():
    """Timer context manager using @contextmanager."""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"Elapsed time: {elapsed:.4f} seconds")

with timer():
    time.sleep(0.1)
    print("Doing work...")


# ----------------------------------------------------------------------------
# 4.3 Exception Suppression with @contextmanager
# ----------------------------------------------------------------------------
print("\n--- Exception Suppression with @contextmanager ---")

@contextmanager
def suppress_exceptions(*exceptions):
    """Suppress specific exceptions."""
    try:
        yield
    except exceptions:
        pass  # Suppress exceptions

with suppress_exceptions(ValueError, TypeError):
    # raise ValueError("This will be suppressed")
    print("No exception raised")


# ----------------------------------------------------------------------------
# 4.4 redirect_stdout
# ----------------------------------------------------------------------------
print("\n--- redirect_stdout ---")

from io import StringIO

# Redirect stdout to a string
output = StringIO()
with redirect_stdout(output):
    print("This goes to StringIO")
    print("Not to console")

print(f"Captured output: {output.getvalue()}")


# ----------------------------------------------------------------------------
# 4.5 redirect_stderr
# ----------------------------------------------------------------------------
print("\n--- redirect_stderr ---")

error_output = StringIO()
with redirect_stderr(error_output):
    import sys
    print("Error message", file=sys.stderr)

print(f"Captured error: {error_output.getvalue()}")


# ----------------------------------------------------------------------------
# 4.6 suppress (Built-in)
# ----------------------------------------------------------------------------
print("\n--- suppress (Built-in) ---")

# Suppress specific exceptions
with suppress(FileNotFoundError, ValueError):
    # This would raise FileNotFoundError, but it's suppressed
    # open('nonexistent.txt')
    print("Exception suppressed if it occurred")


# ----------------------------------------------------------------------------
# 4.7 closing (for objects with close() method)
# ----------------------------------------------------------------------------
print("\n--- closing (for objects with close() method) ---")

from contextlib import closing
from urllib.request import urlopen

# closing() ensures close() is called
# Example with URL (commented out to avoid network dependency)
# with closing(urlopen('http://www.python.org')) as page:
#     content = page.read()
#     print("Page content retrieved")


# ----------------------------------------------------------------------------
# 4.8 ExitStack (Multiple Context Managers)
# ----------------------------------------------------------------------------
print("\n--- ExitStack (Multiple Context Managers) ---")

from contextlib import ExitStack

# Dynamically manage multiple context managers
with ExitStack() as stack:
    files = [
        stack.enter_context(open(f'file{i}.txt', 'w'))
        for i in range(3)
    ]
    for i, f in enumerate(files):
        f.write(f"Content {i}\n")
    print("All files written")
# All files automatically closed


# ============================================================================
# 5. ADVANCED PATTERNS
# ============================================================================

print("\n" + "=" * 70)
print("5. ADVANCED PATTERNS")
print("=" * 70)

# ----------------------------------------------------------------------------
# 5.1 Context Manager Returning Different Object
# ----------------------------------------------------------------------------
print("\n--- Context Manager Returning Different Object ---")

class ResourceManager:
    """Context manager that returns a different object."""
    
    def __init__(self, resource_name):
        self.resource_name = resource_name
        self.resource = None
    
    def __enter__(self):
        print(f"Acquiring {self.resource_name}")
        # Return a wrapper object instead of self
        class ResourceWrapper:
            def __init__(self, name):
                self.name = name
            
            def use(self):
                return f"Using {self.name}"
        
        self.resource = ResourceWrapper(self.resource_name)
        return self.resource
    
    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Releasing {self.resource_name}")
        return False

with ResourceManager("Database") as resource:
    print(resource.use())


# ----------------------------------------------------------------------------
# 5.2 Reusable Context Manager
# ----------------------------------------------------------------------------
print("\n--- Reusable Context Manager ---")

class ReusableLock:
    """Context manager that can be reused."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.acquired = False
    
    def __enter__(self):
        if not self.acquired:
            self.lock.acquire()
            self.acquired = True
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        # Don't release - allow reuse
        return False
    
    def release(self):
        """Manually release the lock."""
        if self.acquired:
            self.lock.release()
            self.acquired = False

lock = ReusableLock()
with lock:
    print("First use")
with lock:
    print("Reused")
lock.release()


# ----------------------------------------------------------------------------
# 5.3 Nested Context Managers
# ----------------------------------------------------------------------------
print("\n--- Nested Context Managers ---")

@contextmanager
def outer_context():
    print("Entering outer context")
    try:
        yield
    finally:
        print("Exiting outer context")

@contextmanager
def inner_context():
    print("Entering inner context")
    try:
        yield
    finally:
        print("Exiting inner context")

with outer_context():
    print("In outer context")
    with inner_context():
        print("In inner context")


# ============================================================================
# 6. PRACTICE EXERCISES
# ============================================================================

print("\n" + "=" * 70)
print("6. PRACTICE EXERCISES")
print("=" * 70)

# Exercise 1: File Manager with Error Handling
print("\n--- Exercise 1: File Manager with Error Handling ---")

class SafeFileManager:
    """File manager that handles errors gracefully."""
    
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        try:
            self.file = open(self.filename, self.mode)
            return self.file
        except FileNotFoundError:
            print(f"File {self.filename} not found")
            return None
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()
        return False  # Don't suppress exceptions

with SafeFileManager('example.txt', 'r') as f:
    if f:
        content = f.read()
        print(f"Content: {content}")


# Exercise 2: Performance Timer
print("\n--- Exercise 2: Performance Timer ---")

@contextmanager
def performance_timer(operation_name):
    """Timer that logs performance metrics."""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"{operation_name} took {elapsed:.4f} seconds")

with performance_timer("Data processing"):
    time.sleep(0.05)
    print("Processing data...")


# Exercise 3: Temporary Environment Variable
print("\n--- Exercise 3: Temporary Environment Variable ---")

@contextmanager
def temporary_env_var(key, value):
    """Temporarily set an environment variable."""
    import os
    old_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value

# Example usage
with temporary_env_var('TEST_VAR', 'test_value'):
    print(f"TEST_VAR = {os.environ.get('TEST_VAR')}")
print(f"After context: TEST_VAR = {os.environ.get('TEST_VAR')}")


# Exercise 4: Transaction-like Context Manager
print("\n--- Exercise 4: Transaction-like Context Manager ---")

class Transaction:
    """Simple transaction context manager."""
    
    def __init__(self):
        self.actions = []
        self.committed = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # No exception - commit
            self.commit()
        else:
            # Exception - rollback
            self.rollback()
        return False
    
    def add_action(self, action):
        """Add an action to the transaction."""
        self.actions.append(action)
    
    def commit(self):
        """Commit all actions."""
        print("Committing transaction:")
        for action in self.actions:
            print(f"  - {action}")
        self.committed = True
    
    def rollback(self):
        """Rollback all actions."""
        print("Rolling back transaction")
        self.actions.clear()

with Transaction() as txn:
    txn.add_action("Update user")
    txn.add_action("Send email")
    print("Transaction completed successfully")


# ============================================================================
# 7. COMMON INTERVIEW QUESTIONS
# ============================================================================

print("\n" + "=" * 70)
print("7. COMMON INTERVIEW QUESTIONS")
print("=" * 70)

# Q1: What is a context manager?
print("\n--- Q1: What is a context manager? ---")
print("""
A context manager is an object that defines what happens when you enter
and exit a 'with' statement. It ensures proper setup and cleanup of resources,
even if exceptions occur. It implements __enter__ and __exit__ methods.
""")

# Q2: Why use context managers?
print("\n--- Q2: Why use context managers? ---")
print("""
- Automatic resource cleanup (files, locks, connections)
- Exception-safe resource management
- Cleaner code than try-finally blocks
- Ensures cleanup code always runs
- Prevents resource leaks
""")

# Q3: What does __exit__ return?
print("\n--- Q3: What does __exit__ return? ---")
print("""
__exit__ should return:
- False (or None): Propagate exceptions (normal behavior)
- True: Suppress exceptions (use carefully)

The return value determines whether exceptions are propagated or suppressed.
""")

# Q4: What's the difference between class-based and @contextmanager?
print("\n--- Q4: Class-based vs @contextmanager ---")
print("""
Class-based:
- Implement __enter__ and __exit__ methods
- More explicit, easier to understand
- Better for complex state management

@contextmanager:
- Generator function with yield
- More concise for simple cases
- Less boilerplate code
- yield separates setup and cleanup
""")

# Q5: Can you nest context managers?
print("\n--- Q5: Nesting Context Managers ---")
print("""
Yes! You can nest context managers:
- Multiple 'with' statements
- Multiple context managers in one 'with': with cm1() as x, cm2() as y:
- ExitStack for dynamic management

They are exited in reverse order (LIFO - Last In First Out).
""")


# ============================================================================
# 8. KEY TAKEAWAYS FOR INTERVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("8. KEY TAKEAWAYS FOR INTERVIEWS")
print("=" * 70)

print("""
1. CONTEXT MANAGER BASICS:
   - Use 'with' statement for resource management
   - Ensures __exit__ is always called (even on exceptions)
   - Cleaner than try-finally blocks
   - Automatic resource cleanup

2. CLASS-BASED CONTEXT MANAGERS:
   - Implement __enter__() and __exit__() methods
   - __enter__() returns value assigned to 'as' variable
   - __exit__() receives exception info (exc_type, exc_value, traceback)
   - Return False to propagate exceptions, True to suppress

3. @contextmanager DECORATOR:
   - Convert generator function to context manager
   - Code before yield is setup (like __enter__)
   - Code after yield is cleanup (like __exit__)
   - yield value is returned to 'as' variable
   - More concise for simple cases

4. BUILT-IN CONTEXT MANAGERS:
   - open(): File handling
   - threading.Lock(): Thread synchronization
   - contextlib.suppress(): Suppress exceptions
   - contextlib.redirect_stdout/stderr: Redirect output
   - contextlib.closing(): Ensure close() is called

5. COMMON PATTERNS:
   - File operations: with open() as f:
   - Locks: with lock:
   - Database connections: with db.connect() as conn:
   - Timers: with Timer():
   - Exception suppression: with suppress(Exception):

6. BEST PRACTICES:
   - Always use 'with' for file operations
   - Use context managers for any resource that needs cleanup
   - Don't suppress exceptions unless necessary
   - Use @contextmanager for simple cases
   - Use class-based for complex state management

7. ADVANTAGES:
   - Prevents resource leaks
   - Exception-safe
   - Cleaner code
   - Explicit resource management
   - Pythonic way to handle resources
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("All examples executed successfully!")
    print("=" * 70)
