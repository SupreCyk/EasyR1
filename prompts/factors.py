FACTORS = """
1. correctness_numeric:
   - Extract the final answer from the model output: if the output contains \boxed{...}, use the content inside \boxed{} as the final answer; otherwise, identify the final numerical or algebraic answer from the model's response. Compare this extracted answer with the ground truth answer. 
   -Attention: Even if the model's answer and the reference answer look different in form, make sure to carefully check if they are actually mathematically equivalent by evaluating their numerical values or algebraic equivalence. Take extra care to determine true equivalence, not just superficial similarity or difference in format.
  - Score must be 0 or 1.

2. visual_interpretation
   - Correctly interprets diagram elements (numbers, lines, angles) without hallucinating.
   - Score: {0, 0.5, 1}.

3. math_validity
   - Uses valid mathematical principles.
   - Score: {0, 0.5, 1}.

4. instruction_following
   - Whether the answer directly responds to the question.
   - Score: {0, 0.5, 1}.

5. expression_format
   - Answer is a clean numeric or algebraic expression with minimal extra text.
   - Score: {0, 0.5, 1}.
"""