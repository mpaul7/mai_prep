"""
Python Data Structures Example: lists, dicts, sets, tuples, stacks/queues, heap.

Covers:
- Common operations on built-in structures
- Comprehensions and generators
- Sorting with custom keys
- Frequency counts and grouping
- Stack (list), queue (deque), priority queue (heapq)
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import heapq
from typing import Dict, Iterable, List, Tuple


def list_operations(nums: List[int]) -> List[int]:
    print(nums)
    squared = [n * n for n in nums]
    print(squared)
    evens = [n for n in squared if n % 2 == 0]
    print(evens)
    return sorted(evens, reverse=True)


def dict_group_by_first_letter(words: Iterable[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    print(words)
    
    print(groups)
    for w in words:
        key = w[0].lower()
        groups[key].append(w)
    # sort each group's values
    for k in groups:
        groups[k].sort()
    return dict(groups)


def top_k_frequent(words: Iterable[str], k: int) -> List[Tuple[str, int]]:
    counts = Counter(words)
    # Sort by frequency desc, then word asc
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:k]


def stack_and_queue_demo(values: Iterable[int]) -> Tuple[List[int], List[int]]:
    stack: List[int] = []
    queue: deque[int] = deque()

    for v in values:
        stack.append(v)  # push
        queue.append(v)  # enqueue

    popped_stack: List[int] = []
    while stack:
        popped_stack.append(stack.pop())  # LIFO

    dequeued: List[int] = []
    while queue:
        dequeued.append(queue.popleft())  # FIFO

    return popped_stack, dequeued


def priority_queue_demo(pairs: Iterable[Tuple[int, str]]) -> List[Tuple[int, str]]:
    # Min-heap ordered by the first element of the pair
    heap: List[Tuple[int, str]] = []
    for pr, item in pairs:
        heapq.heappush(heap, (pr, item))
    out: List[Tuple[int, str]] = []
    while heap:
        out.append(heapq.heappop(heap))
    return out


def main() -> None:
    nums = [5, 3, 2, 2, 7, 1]
    print("list_operations:", list_operations(nums))

    words = ["Apple", "apricot", "banana", "Blue", "black", "cherry"]
    print("dict_group_by_first_letter:", dict_group_by_first_letter(words))

    words2 = ["a", "b", "a", "c", "b", "a", "d", "d", "d", "d"]
    print("top_k_frequent:", top_k_frequent(words2, 2))

    popped, dequeued = stack_and_queue_demo([1, 2, 3, 4])
    print("stack pop order:", popped)
    print("queue dequeue order:", dequeued)

    pq = priority_queue_demo([(3, "low"), (1, "urgent"), (2, "normal")])
    print("priority_queue order:", pq)


if __name__ == "__main__":
    main()


