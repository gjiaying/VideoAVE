import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import ast
import requests
from PIL import Image
from io import BytesIO
import time
import torch
import gc
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load full data
csv_path = "test_all_image.csv"
df = pd.read_csv(csv_path)

# Parse aspects into dict
def parse_aspects(aspects_str):
    try:
        aspects_str = aspects_str.replace("'", '"')
        return ast.literal_eval(aspects_str)
    except:
        return {}

df['aspects_dict'] = df['aspects'].apply(parse_aspects)

# Define image loading function
def load_image_from_url(url, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"[Attempt {attempt+1}] Failed to load {url}: {e}")
    return None

# Define generation function
def generate(model, processor, images, prompt, max_tokens=512):
    if not isinstance(images, list):
        images = [images]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            *[{"type": "image", "image": img} for img in images if img is not None]
        ]}
    ]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        generated = outputs[0][inputs.input_ids.shape[1]:]
        output_text = processor.decode(generated, skip_special_tokens=True)

        try:
            parsed = ast.literal_eval("{" + output_text.strip().strip(',') + "}")
        except:
            parsed = {"raw_response": output_text}
        return parsed

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        gc.collect()
        print("[OOM ERROR] Skipping due to CUDA OOM")
        return {"error": "CUDA Out of Memory"}

# Chunked processing function
def run_experiment(df, mode, chunk_size=50):
    start_time = time.time()
    all_results = []
    num_chunks = (len(df) + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        print(f"\nLoading model for {mode} | Chunk {chunk_idx+1}/{num_chunks}")
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=128*28*28,
            max_pixels=384*28*28
        )

        chunk = df.iloc[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        results = []

        for row in tqdm(chunk.itertuples(), total=len(chunk), desc=f"{mode} | Chunk {chunk_idx+1}/{num_chunks}"):
            try:
                urls = ast.literal_eval(row.hi_res)
                if isinstance(urls, str):
                    urls = [urls]
            except:
                urls = [row.hi_res]

            if not isinstance(urls, list):
                urls = [urls]

            images = [load_image_from_url(url) for url in urls if url and isinstance(url, str) and load_image_from_url(url) is not None]
            if not images and "title" not in mode:
                results.append({"error": "[No valid image]"})
                continue

            if mode == "attr-single":
                attributes = list(row.aspects_dict.keys())
                prompt = (
                    "You are a professional product analyst specializing in attribute extraction for leading e-commerce platforms. "
                    f"Your task is to extract the values of the following attributes for the product in this image: {', '.join(attributes)}. "
                    "Answer it in this format only: 'attribute1': 'attribute1_value', 'attribute2': 'attribute2_value', ... "
                    "Choose one specific value for each attribute."
                )
                result = generate(model, processor, images[0], prompt)
                results.append(result)

            elif mode == "attr-multi":
                attributes = list(row.aspects_dict.keys())
                prompt = (
                    "You are a professional product analyst specializing in attribute extraction for leading e-commerce platforms. "
                    f"Your task is to extract the values of the following attributes for the product in these images: {', '.join(attributes)}. "
                    "Answer it in this format only: 'attribute1': 'attribute1_value', 'attribute2': 'attribute2_value', ... "
                    "Choose one specific value for each attribute."
                )
                result = generate(model, processor, images, prompt)
                results.append(result)

            elif mode == "gen-single":
                prompt = (
                    "You are a professional product analyst specializing in attribute extraction for leading e-commerce platforms. "
                    "Your task is to extract the values of all relevant attributes for the product in this image. "
                    "Answer it in this format only: 'attribute1': 'attribute1_value', 'attribute2': 'attribute2_value', ... "
                    "Choose one specific value for each attribute."
                )
                result = generate(model, processor, images[0], prompt)
                results.append(result)

            elif mode == "gen-multi":
                prompt = (
                    "You are a professional product analyst specializing in attribute extraction for leading e-commerce platforms. "
                    "Your task is to extract the values of all relevant attributes for the product in these images. "
                    "Answer it in this format only: 'attribute1': 'attribute1_value', 'attribute2': 'attribute2_value', ... "
                    "Choose one specific value for each attribute."
                )
                result = generate(model, processor, images, prompt)
                results.append(result)

            elif mode == "title-attr":
                attributes = list(row.aspects_dict.keys())
                prompt = (
                    "You are a professional product analyst. Based on the following product title, extract the values of the following attributes: "
                    f"{', '.join(attributes)}. Title: {row.parent_title}. "
                    "Answer it in this format only: 'attribute1': 'attribute1_value', 'attribute2': 'attribute2_value', ... "
                    "Choose one specific value for each attribute."
                )
                result = generate(model, processor, [], prompt)
                results.append(result)

            elif mode == "title-gen":
                prompt = (
                    "You are a professional product analyst. Based on the following product title, extract all relevant product attributes. "
                    f"Title: {row.parent_title}. "
                    "Answer it in this format only: 'attribute1': 'attribute1_value', 'attribute2': 'attribute2_value', ... "
                    "Choose one specific value for each attribute."
                )
                result = generate(model, processor, [], prompt)
                results.append(result)

            torch.cuda.empty_cache()
            gc.collect()

        chunk_df = pd.DataFrame({"product_id": chunk["product_id"].values, "output": results})
        all_results.append(chunk_df)

        # cleanup
        del model
        del processor
        torch.cuda.empty_cache()
        gc.collect()

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(f"all_outputs_{mode}.csv", index=False)
    duration = time.time() - start_time
    print(f"Mode: {mode}, Time taken: {duration:.2f} seconds")
    return final_df, duration

# Modes to run
modes = ["attr-multi", "attr-single", "gen-multi", "gen-single", "title-attr", "title-gen"]
summary = []

for mode in modes:
    _, duration = run_experiment(df, mode)
    summary.append((mode, duration))

# Print summary
print("\n=== Summary of Timings ===")
for mode, duration in summary:
    print(f"{mode}: {duration:.2f} seconds")
