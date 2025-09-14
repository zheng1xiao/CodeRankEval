IN_CONTEXT_EXAMPLE = """
Strictly evaluate the Python code generated for the given problem using the following scoring criteria:

Scoring Criteria (score: 1-5 points):

Score 1 - Very Poor:
Code fails to run or is unrelated to the problem.
Contains serious syntax errors.
The algorithm is fundamentally flawed.
Fails on basic test inputs.

Score 2 - Below Average:
Code runs but has major flaws.
Can only handle a subset of test cases.
Contains significant efficiency issues.
The code structure is messy.

Score 3 - Mostly Correct:
Code solves the problem to some extent, but may have minor issues.
Passes most basic test cases but might fail in certain edge cases.
May lack proper error handling.
Code structure is somewhat clear but could be improved.

Score 4 - Good:
Code is relatively elegant and efficient.
Includes fairly complete error handling.
Considers some edge cases.
Code readability is good, with some modular design.

Score 5 - Excellent:
Code is concise, elegant, and highly efficient.
Includes comprehensive error handling.
Thoroughly considers edge cases.
Employs optimal algorithms with well-structured modular design.


Instructions:

1.Provide a detailed analysis of the code quality.
2.Specify the corresponding Score (1-5).
"""

IN_CONTEXT_EXAMPLE_LISTWISE = """
Strictly evaluate the Python code generated for the given problem by ranking them from best to worst based on the following evaluation criteria:

### **Evaluation Criteria:**
1. **Correctness:** Does the code solve the problem correctly?
2. **Efficiency:** Does the implementation use an optimal approach in terms of time and space complexity?
3. **Readability & Maintainability:** Is the code well-structured, readable, and easy to understand?
4. **Best Practices:** Does the implementation follow Pythonic style, proper error handling, and modular design? """


GENERATE_CONTEXT_EXAMPLE = """
Below are the scoring criteria for the Python code generation task:"

Scoring Criteria (score: 1-5 points):

Score 1 - Very Poor:
Code fails to run or is unrelated to the problem.
Contains serious syntax errors.
The algorithm is fundamentally flawed.
Fails on basic test inputs.

Score 2 - Below Average:
Code runs but has major flaws.
Can only handle a subset of test cases.
Contains significant efficiency issues.
The code structure is messy.

Score 3 - Mostly Correct:
Code solves the problem to some extent, but may have minor issues.
Passes most basic test cases but might fail in certain edge cases.
May lack proper error handling.
Code structure is somewhat clear but could be improved.

Score 4 - Good:
Code is relatively elegant and efficient.
Includes fairly complete error handling.
Considers some edge cases.
Code readability is good, with some modular design.

Score 5 - Excellent:
Code is concise, elegant, and highly efficient.
Includes comprehensive error handling.
Thoroughly considers edge cases.
Employs optimal algorithms with well-structured modular design.

"""

POINTWISE_SYSTEM_PROMPT = (
    "As a senior Python code quality assessor, you possess extensive programming expertise and rigorous evaluation standards. "
    "Your responsibility is to ensure code adheres to best practices, including code readability, performance optimization, error handling, and design patterns. "
    "Please provide detailed assessments based on your professional judgment."
)

LISTWISE_SYSTEM_PROMPT = (
    "As a senior Python code quality assessor, you possess extensive programming expertise and rigorous evaluation standards. "
    "Your responsibility is to rank multiple solutions based on best practices, correctness, efficiency, readability, and maintainability. "
    "Ensure that your ranking is precise and well-justified, with clear reasoning for the ordering."
)