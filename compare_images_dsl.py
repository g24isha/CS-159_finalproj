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
    " Examples of good questions:\n"
    "- What is the color of the umbrella?\n"
    "- How many people are on the balcony?\n"
    "- Is there a car visible in the lower right corner?\n"
    "- What is the shape of the object near the door?\n\n"
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
img1_path = "assets/difflive1.png"
img2_path = "assets/difflive2.png"
diff_path = "assets/difference_heatmap.png"

questions = get_comparison_questions(img1_path, img2_path, diff_path)
execute_visprog_symbolic(img1_path, img2_path, questions)
