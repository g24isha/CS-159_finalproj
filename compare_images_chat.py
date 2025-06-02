from PIL import Image
import base64
import requests
import json
import os
from io import BytesIO
from config import OPENAI_API_KEY


# === Prompt for generating localized questions ===
GPT_TASK_PROMPT = (
    "You are given two images of the same scene and a difference heatmap. "
    "The difference heatmap is a binary image where the difference between the two images is highlighted in white. "
    "Generate AS MANY *specific and visually testable* questions that help identify differences between them as you can. "
    "Use the difference heatmap to generate questions that focus on the differences between the two images. "
    "Cover the following categories:\n"
    "- Presence or absence of objects (e.g. 'Is there a boat?')\n"
    "- Quantity (e.g. 'How many cars are there?')\n"
    "- Color (e.g. 'What is the color of the shirt?')\n"
    "- Shape or orientation (e.g. 'What shape is the object? Is it upright or tilted?')\n"
    "- Action or behavior (e.g. 'Is anyone walking? Is the dog sitting?')\n\n"
    "Write each question so it can be answered **independently** for each image — do not compare directly (e.g. not 'Which image has more trees?').\n\n"
    "Do not combine questions. Do not repeat the same wording.\n\n"
    "Each question should focus on a specific object, color, action, or spatial detail. "
    "Questions should have one word answers. "
    "Respond only with a JSON array of strings (no explanation, no Markdown)."
)


# === Util: base64 encode an image file ===
def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# === Call GPT-4o to generate localized questions from diff map ===
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
        elif content.strip().startswith("```"):
            stripped = content.strip().strip("```").strip()
            return json.loads(stripped)
        else:
            return json.loads(content.strip())
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        print("Falling back to default questions.")
        return [
            "What objects are visible?",
            "What are the main colors?",
            "What is the cat holding?",
            "What is the shape of the cake?",
        ]


# === Call GPT-4o VQA for a single image and question ===
def vqa_with_gpt4o(image_pil, question):
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "ANSWER IN ONE WORD OR TWO ONLY. Like you are a robot."},
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]
        }],
        "max_tokens": 100
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"GPT-4o VQA failed: {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()


# === Run question-by-question comparison using GPT-4o for answers ===
def execute_visprog_comparison(img1_path, img2_path, questions):
    difference_counter = 0
    img_left = Image.open(img1_path).convert("RGB")
    img_right = Image.open(img2_path).convert("RGB")
    img_left.thumbnail((640, 640), Image.Resampling.LANCZOS)
    img_right.thumbnail((640, 640), Image.Resampling.LANCZOS)

    print("\n🧠 GPT-Generated Questions & GPT-4o VQA Results\n" + "=" * 60)
    for i, question in enumerate(questions, 1):
        print(f"\n ->> Question {i}: {question}")

        try:
            left_ans = vqa_with_gpt4o(img_left, question)
            right_ans = vqa_with_gpt4o(img_right, question)

            print(f"  LEFT : {left_ans}")
            print(f"  RIGHT: {right_ans}")

            norm = lambda s: str(s).strip().lower()
            if norm(left_ans) != norm(right_ans):
                difference_counter += 1
            print(f"  ➤ Different? → {'Yes' if norm(left_ans) != norm(right_ans) else 'No'}")
        except Exception as e:
            print(f"  Error during GPT-4o VQA: {e}")
    print("TOTAL DIFFERENCES FOUND:", difference_counter)


# === Main pipeline ===
def compare_images(img1_path, img2_path, diff_path):
    if not (os.path.exists(img1_path) and os.path.exists(img2_path) and os.path.exists(diff_path)):
        raise FileNotFoundError("One or more input files do not exist.")

    print("📡 Getting localized comparison questions from GPT-4o...")
    questions = get_comparison_questions(img1_path, img2_path, diff_path)

    print("\n📝 Questions to test:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")

    execute_visprog_comparison(img1_path, img2_path, questions)


# === Entrypoint ===
if __name__ == "__main__":
    img1_path = "assets/difflive1.png"
    img2_path = "assets/difflive2.png"
    diff_path = "assets/difference_heatmap.png"

    try:
        compare_images(img1_path, img2_path, diff_path)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure:")
        print("1. Your OpenAI API key is valid in config.py")
        print("2. The three image files exist")
        print("3. You are connected to the internet")
