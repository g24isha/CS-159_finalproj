from PIL import Image
from engine.utils import ProgramInterpreter
from engine.step_interpreters import register_step_interpreters 

import base64
import requests
import json
import os
from config import OPENAI_API_KEY

register_step_interpreters()

GPT_TASK_PROMPT = (
    "You are given two images of the same scene and a difference heatmap. "
    "The difference heatmap is a binary image where white pixels indicate regions of change. "
    "Your task is to generate as many **independent, concrete, and visually answerable** questions as possible that help identify differences between the images.\n\n"
    "PURPOSE: The questions will be use to generate more specific questions that will be automatically mapped to symbolic reasoning modules (like VQA, FIND, COUNT, EXISTS). "
    "Your job is to phrase questions that can be answered using those operations, focused ONLY on changed regions.\n\n"
    " Guidelines:\n"
    "- Focus on **visual attributes** (color, count, presence, shape, size, material, visibility, texture).\n"
    "- Write **one question per object** or concept — split compound questions.\n"
    "- Avoid vague or abstract questions (like 'Is it pretty?').\n"
    "- Do NOT refer to both images at once (no comparisons like 'Is it different?').\n"
    "- DO NOT write symbolic code or refer to modules like FIND or VQA.\n"
    "- DO NOT reference the heatmap or the fact that there are two images.\n"
    "- Avoid **overly specific references** like exact positions ('bottom right') or assumed object names — use general object terms where possible (e.g., 'the object on the tray', 'the figure on the shelf').\n"
    "- Make sure the question has a **factual, short answer** (e.g., yes/no, number, or single word).\n\n"
    " Examples of good questions:\n"
    "- What is the color of the umbrella?\n"
    "- How many people are on the balcony?\n"
    "- Is there a car visible in the lower right corner?\n"
    "- What is the shape of the object near the door?\n\n"
    " Examples of overly specific questions to avoid:\n"
    "- What is the color of the liquid in the cup on the top shelf?\n"
    "- How many red dolls are on the left side of the middle shelf?\n\n"
    "Return ONLY a JSON array of questions like:\n"
    "[\"What color is the umbrella?\", \"How many dogs are there?\", ...]"
)



def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_comparison_questions(img1_path, img2_path, diff_path):
    img1_b64 = encode_image_to_base64(img1_path)
    img2_b64 = encode_image_to_base64(img2_path)
    diff_b64 = encode_image_to_base64(diff_path)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": GPT_TASK_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{diff_b64}"}}
                ]
            }
        ]
    }

    res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if res.status_code != 200:
        raise RuntimeError(f"OpenAI API error: {res.status_code} - {res.text}")

    content = res.json()['choices'][0]['message']['content']
    print("\n🧾 Raw GPT Response:")
    print(content)

    try:
        if content.strip().startswith("```json"):
            stripped = content.strip().strip("```json").strip("```").strip()
            return json.loads(stripped)
        return json.loads(content.strip())
    except Exception as e:
        print(f"Failed to parse GPT response: {e}")
        return []

def get_follow_up_qs(img1_path, img2_path, diff_path, parent_q):
    FOLLOW_UP_PROMPT = f"""
    "You are given two images of the same scene and a difference heatmap. "
    "The difference heatmap is a binary image where white pixels indicate regions of change. "
    "You are also given these high-level questions about differences between the images:
    {json.dumps(parent_q, indent=2)}"
    "Your task is to generate as many **specific, independent, concrete, and visually answerable** sub-questions for each high-level question as possible that help identify each minute difference between the images.\n\n"
    "PURPOSE: The questions will be automatically mapped to symbolic reasoning modules (like VQA, FIND, COUNT, EXISTS). "
    "Your job is to phrase questions that can be answered using those operations, focused ONLY on changed regions.\n\n"
    " Guidelines:\n"
    "- Focus on **visual attributes** (color, count, presence, shape, size, material, visibility, texture).\n"
    "- Write **one question per object** or concept — split compound questions.\n"
    "- Avoid vague or abstract questions (like 'Is it pretty?').\n"
    "- Do NOT refer to both images at once (no comparisons like 'Is it different?').\n"
    "- DO NOT write symbolic code or refer to modules like FIND or VQA.\n"
    "- DO NOT reference the heatmap or the fact that there are two images.\n"
    "- Make sure the question has a **factual, short answer** (e.g., yes/no, number, or single word).\n\n"
    "Refinement strategies: "
       " - Add spatial specificity (general → specific location) "
       " - Add attribute specificity (object → object + color/size/state) "
        "- Add quantitative precision (things → exact counts)"
        "- Break compound questions into atomic parts"
        
       " Examples:"
       " Level 1: Are there any vehicles visible?"
       " Level 2: [Is there a car in the image?, "
                  "Is there a truck in the image?",
                  "Are there any motorcycles visible?]"
        
       " Level 1: What is in the upper portion?"
        "Level 2: [What objects are in the upper left quadrant?,"
                  "What is in the upper right corner? ,"
                  "What is at the very top of the image?]"
                  
        "Level 2: Is there a car in the image?"
        "Level 3: [Is there a red car visible?,"
                  "Is there a car in the foreground?,"
                  "Is there a parked car?]" 
    "Return a JSON object mapping each parent question to its refined versions:"
    "{{"
        "parent_question_1": ["refined_1", "refined_2", "refined_3"],
        "parent_question_2": ["refined_1", "refined_2"]
   "}}"
    
"""
    img1_b64 = encode_image_to_base64(img1_path)
    img2_b64 = encode_image_to_base64(img2_path)
    diff_b64 = encode_image_to_base64(diff_path)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": FOLLOW_UP_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{diff_b64}"}}
                ]
            }
        ]
    }

    res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if res.status_code != 200:
        raise RuntimeError(f"OpenAI API error: {res.status_code} - {res.text}")

    content = res.json()['choices'][0]['message']['content']
    print("\n🧾 Raw GPT Response:")
    print(content)

    try:
        if content.strip().startswith("```json"):
            stripped = content.strip().strip("```json").strip("```").strip()
            return json.loads(stripped)
        return json.loads(content.strip())
    except Exception as e:
        print(f"Failed to parse GPT response: {e}")
        return []

    

def clean_program(content):
    lines = content.strip().split("\n")
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def generate_symbolic_program(question, image_side):
    prompt = f"""
Generate a VisProg program to answer the question. Use ONLY these functions:
- VQA(image=..., question=...)
- FIND(image=..., object=...)
- COUNT(region=...)
- EXISTS(region=...)
- RESULT(var=...)

IMPORTANT:
- If the question can be answered using VQA, use VQA(image=..., question=...) first.
- Each question should have only one object as subject; Break queries with multiple subject 
    down to multiple questions. 
- Make sure each object has its own question.
- Always question the more significant object first. 
- Format each line as: output_var = FUNCTION(arg1=value1, arg2=value2)
- The final line must always be: result = RESULT(var=...)
- Do not use markdown.
- NO Python code like comparisons (==) or conditionals (if)

Question: {question}
Image: {image_side}
""".strip()

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.1
    }

    res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if res.status_code != 200:
        raise RuntimeError(f"OpenAI error: {res.status_code} - {res.text}")

    return clean_program(res.json()['choices'][0]['message']['content'])

def execute_visprog_symbolic_followup(img1_path, img2_path, questions):
    interpreter = ProgramInterpreter(dataset='nlvr')
    img_left = Image.open(img1_path).convert("RGB")
    img_right = Image.open(img2_path).convert("RGB")
    img_left.thumbnail((640, 640), Image.Resampling.LANCZOS)
    img_right.thumbnail((640, 640), Image.Resampling.LANCZOS)
    state = {"LEFT": img_left, "RIGHT": img_right}

    difference_counter = 0
    print("\n🔎 Executing Symbolic Programs via VisProg\n" + "="*60)
    for i, (parent_question, follow_ups) in enumerate(questions.items(), 1):
        print(f"\n→ Question {i}: {parent_question}")
        for j, q in enumerate(follow_ups):
            letter = chr(97 + j)
            print(f"\n→ Question {i}{letter}: {q}")
            try:
                prog_template = generate_symbolic_program(q, "IMAGE_PLACEHOLDER")
                prog_L = prog_template.replace("IMAGE_PLACEHOLDER", "LEFT")
                prog_R = prog_template.replace("IMAGE_PLACEHOLDER", "RIGHT")
                # prog_L = generate_symbolic_program(q, "LEFT")
                # prog_R = generate_symbolic_program(q, "RIGHT")

                print("[LEFT DSL]")
                print(prog_L)
                left_ans, _, _ = interpreter.execute(prog_L, state, inspect=True)

                print("\n[RIGHT DSL]")
                print(prog_R)
                right_ans, _, _ = interpreter.execute(prog_R, state, inspect=True)

                print(f"\nLEFT : {left_ans}")
                print(f"RIGHT: {right_ans}")
                norm = lambda s: str(s).strip().lower()
                print(f"➤ Different? → {'Yes' if norm(left_ans) != norm(right_ans) else 'No'}")
                if norm(left_ans) != norm(right_ans):
                    difference_counter += 1

            except Exception as e:
                print(f"Error: {e}")

    print("\nTOTAL DIFFERENCES FOUND:", difference_counter)
    
def execute_visprog_symbolic(img1_path, img2_path, questions):
    interpreter = ProgramInterpreter(dataset='nlvr')
    img_left = Image.open(img1_path).convert("RGB")
    img_right = Image.open(img2_path).convert("RGB")
    img_left.thumbnail((640, 640), Image.Resampling.LANCZOS)
    img_right.thumbnail((640, 640), Image.Resampling.LANCZOS)
    state = {"LEFT": img_left, "RIGHT": img_right}

    difference_counter = 0
    print("\n🔎 Executing Symbolic Programs via VisProg\n" + "="*60)
    for i, question in enumerate(questions, 1):
        print(f"\n→ Question {i}: {question}")
        try:
            prog_L = generate_symbolic_program(question, "LEFT")
            prog_R = generate_symbolic_program(question, "RIGHT")

            print("[LEFT DSL]")
            print(prog_L)
            left_ans, _, _ = interpreter.execute(prog_L, state, inspect=True)

            print("\n[RIGHT DSL]")
            print(prog_R)
            right_ans, _, _ = interpreter.execute(prog_R, state, inspect=True)

            print(f"\nLEFT : {left_ans}")
            print(f"RIGHT: {right_ans}")
            norm = lambda s: str(s).strip().lower()
            print(f"➤ Different? → {'Yes' if norm(left_ans) != norm(right_ans) else 'No'}")
            if norm(left_ans) != norm(right_ans):
                difference_counter += 1

        except Exception as e:
            print(f"Error: {e}")

    print("\nTOTAL DIFFERENCES FOUND:", difference_counter)

# Example usage:
img1_path = "assets/parking_lot1.png"
img2_path = "assets/parking_lot2.png"
diff_path = "assets/difference_heatmap_5.png"

questions = get_comparison_questions(img1_path, img2_path, diff_path)
specific_qs = get_follow_up_qs(img1_path, img2_path, diff_path, questions)
execute_visprog_symbolic_followup(img1_path, img2_path, specific_qs)

"""
    🔎 Executing Symbolic Programs via VisProg
============================================================

→ Question 1: What is the color of the fruit the raccoon is holding?

→ Question 1a: Is the fruit the raccoon is holding yellow?
[LEFT DSL]
fruit_location = FIND(image=LEFT, object="fruit")
is_fruit_yellow = VQA(image=fruit_location, question="Is the fruit yellow?")
result = RESULT(var=is_fruit_yellow)
FIND
/Users/g24isha/Library/CloudStorage/OneDrive-CaliforniaInstituteofTechnology/second year/cs 159/CS-159/cs159_env_new/lib/python3.11/site-packages/transformers/models/owlvit/processing_owlvit.py:233: FutureWarning: `post_process_object_detection` method is deprecated for OwlVitProcessor and will be removed in v5. Use `post_process_grounded_object_detection` instead.
  warnings.warn(
VQA
RESULT

[RIGHT DSL]
raccoon_with_fruit = FIND(image=RIGHT, object="raccoon with fruit")
fruit_color = VQA(image=raccoon_with_fruit, question="What color is the fruit?")
is_yellow = (fruit_color == "yellow")
result = RESULT(var=is_yellow)
FIND
VQA
Error: Invalid step format

→ Question 1b: Is the fruit the raccoon is holding red?
[LEFT DSL]
fruit_location = FIND(image=LEFT, object="fruit")
is_fruit_red = VQA(image=fruit_location, question="Is this red?")
result = RESULT(var=is_fruit_red)
FIND
VQA
RESULT

[RIGHT DSL]
fruit_location = FIND(image=RIGHT, object="fruit")
is_fruit_red = VQA(image=fruit_location, question="Is this red?")
result = RESULT(var=is_fruit_red)
FIND
VQA
RESULT

LEFT : no
RIGHT: yes
➤ Different? → Yes

→ Question 2: How many fruits are there on the tree?

→ Question 2a: Is there one fruit on the tree?
[LEFT DSL]
tree_region = FIND(image=LEFT, object="tree")
fruit_on_tree = FIND(image=tree_region, object="fruit")
fruit_count = COUNT(region=fruit_on_tree)
result = RESULT(var=fruit_count)
FIND
FIND
COUNT
RESULT

[RIGHT DSL]
tree_region = FIND(image=RIGHT, object="tree")
fruit_on_tree = FIND(image=tree_region, object="fruit")
fruit_count = COUNT(region=fruit_on_tree)
result = RESULT(var=fruit_count)
FIND
FIND
COUNT
RESULT

LEFT : 92
RIGHT: 57
➤ Different? → Yes

→ Question 2b: Are there two fruits on the tree?
[LEFT DSL]
tree_region = FIND(image=LEFT, object="tree")
fruits_on_tree = COUNT(region=tree_region)
result = EXISTS(region=fruits_on_tree)
FIND
COUNT
EXISTS

[RIGHT DSL]
tree_region = FIND(image=RIGHT, object="tree")
fruits_on_tree = COUNT(region=tree_region)
result = RESULT(var=fruits_on_tree)
FIND
COUNT
RESULT

LEFT : True
RIGHT: 4
➤ Different? → Yes

→ Question 2c: Are there five fruits on the tree?
[LEFT DSL]
tree_region = FIND(image=LEFT, object="tree")
fruits_on_tree = COUNT(region=tree_region)
result = RESULT(var=fruits_on_tree)
FIND
COUNT
RESULT

[RIGHT DSL]
tree_region = FIND(image=RIGHT, object="tree")
fruits_on_tree = COUNT(region=tree_region)
result = RESULT(var=fruits_on_tree)
FIND
COUNT
RESULT

LEFT : 5
RIGHT: 4
➤ Different? → Yes

→ Question 3: What color are the fruits on the tree?

→ Question 3a: Are the fruits on the tree green?
[LEFT DSL]
output_var1 = VQA(image=LEFT, question="Are the fruits on the tree green?") 
result = RESULT(var=output_var1)
VQA
RESULT

[RIGHT DSL]
tree_region = FIND(image=RIGHT, object="tree")
fruits_region = FIND(image=tree_region, object="fruits")
green_fruits = EXISTS(region=fruits_region, color="green")
result = RESULT(var=green_fruits)
FIND
FIND
EXISTS
RESULT

LEFT : yes
RIGHT: True
➤ Different? → Yes

→ Question 3b: Are the fruits on the tree red?
[LEFT DSL]
output_var1 = VQA(image=LEFT, question="Are the fruits on the tree red?") 
result = RESULT(var=output_var1)
VQA
RESULT

[RIGHT DSL]
output_var = VQA(image=RIGHT, question="Are the fruits on the tree red?")
result = RESULT(var=output_var)
VQA
RESULT

LEFT : yes
RIGHT: yes
➤ Different? → No

TOTAL DIFFERENCES FOUND: 5
    """