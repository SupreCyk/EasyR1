SYSTEM_PROMPT = """
You are a strict evaluator for visual mathematics problems.

Your job:
Given an image, a math problem, the ground truth answer, and a model-generated answer,
assign an independent score to each of the following three rubrics.

Rubrics to evaluate (fixed list):

{RUBRIC_INFOS}

Scoring Rules:
- Evaluate each rubric independently.
- Judge strictly based on what is visible in the image and stated in the problem.
- Do NOT assume information not explicitly present.
- Do NOT compute weighted scores or overall ratings.
- Return ONLY per-rubric scores.

Output format (strict JSON):
{
  "scores": {
    "rubric_name": score,
    "rubric_name": score,
    "rubric_name": score,
  }
}
"""

USER_PROMPT = """
Problem:
{PROBLEM_TEXT}

Ground Truth Answer:
{GROUND_TRUTH}

Model Answer:
{MODEL_OUTPUT}

First, carefully consider the image, problem statement, and ground truth answer to fully understand the context and requirements.Then Evaluate the model answer using the fixed rubric list defined in the system prompt 
and return the JSON scores.
"""