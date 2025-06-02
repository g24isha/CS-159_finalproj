import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

class BlipVQA:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(self.device)
        self.model.eval()

    def ask(self, image: Image.Image, question: str) -> str:
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)
