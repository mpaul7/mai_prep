### Hackerrank-style Data Structures Questions (Python)

#### Q1. Anagram Groups
Given a list of words (stdin or CSV column), group anagrams together. Words are lowercase ASCII letters.

Task:
- Return groups as lists sorted alphabetically within each group, and groups sorted by decreasing size then lexicographically by the first element.

Output: one group per line, words separated by a single space.

---

#### Q2. Merge Dictionaries with Sum
Given N dictionaries (JSON objects) with string keys and integer values, merge them by summing values for identical keys.

Task:
- Output a single dictionary where the value for each key is the sum across all inputs. Keys should be sorted ascending in the output.

---

#### Q3. K Most Frequent Items with Tie-Breaks
Given a list of strings and integer K, return the K most frequent items.

Rules:
- Sort primarily by frequency descending, then lexicographically ascending on the string.
- If fewer than K unique items, return all.

Output: CSV with columns `item,frequency` sorted as specified.


