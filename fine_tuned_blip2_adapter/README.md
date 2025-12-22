---
base_model: Salesforce/blip2-opt-2.7b
library_name: peft
license: apache-2.0
tags:
- vision-language-model
- image-captioning
- lora
- peft
- flickr30k
- multimodal-rag
- transformers
---

# BLIP-2 (OPT-2.7b) Fine-Tuned on Flickr30k

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)
![Task](https://img.shields.io/badge/Task-Image%20Captioning-green)

</div>

## Model Summary

This is a **Low-Rank Adaptation (LoRA)** adapter for the **[Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b)** model. 

It has been fine-tuned on the **Flickr30k** dataset to improve the model's ability to generate descriptive, context-rich image captions suitable for **Multimodal Retrieval Augmented Generation (RAG)** systems. By using PEFT (Parameter-Efficient Fine-Tuning), this adapter allows the base model to adapt to the specific linguistic style of Flickr30k without modifying the original pre-trained weights.

- **Developed by:** [Nama Anda / Your Name]
- **Model type:** Vision-Language Model (LoRA Adapter)
- **Language:** English
- **Finetuned from:** Salesforce/blip2-opt-2.7b

## Intended Uses

### Direct Use
- **Image Captioning:** Generating detailed descriptions for images.
- **Visual Context Generation:** Serving as the "Vision Encoder" in Multimodal RAG pipelines to convert visual data into text for LLM reasoning.

### Out-of-Scope Use
- Not intended for OCR (Optical Character Recognition).
- Not intended for medical imaging diagnosis without further domain-specific training.

## How to Get Started

To use this model, you need to load the base BLIP-2 model and then apply this LoRA adapter.

```python
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel

# 1. Load Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# 2. Load Base Model
base_model_id = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(base_model_id)
model = Blip2ForConditionalGeneration.from_pretrained(
    base_model_id, 
    torch_dtype=dtype, 
    device_map="auto"
)

# 3. Load This Fine-Tuned Adapter
adapter_path = "path/to/fine_tuned_blip2_adapter" # Point to this folder
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# 4. Inference
image = Image.open("test_image.jpg").convert('RGB')
inputs = processor(images=image, return_tensors="pt").to(device, dtype)

generated_ids = model.generate(**inputs, max_new_tokens=50)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(f"Generated Caption: {caption}")