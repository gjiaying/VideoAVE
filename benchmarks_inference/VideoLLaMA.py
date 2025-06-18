import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython.display import Markdown, clear_output, display, Video
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
import argparse
import pandas as pd
from tqdm import tqdm


# prompt (provide aspect keys)
def preprocess(df):
    id_to_prompt = {}
    id_to_label = {}

    # Create a prompt that includes the specific attributes for each product
    df['prompt'] = df.apply(lambda row: f"You are a professional product analyst specializing in attribute extraction for leading e-commerce platforms.Your task is to extract the values of the following attributes for the product in this video: {row['attributes']}. Only the values of these attributes should be extracted. Do not include any kinds of explanation. Answer it in this format only: 'attribute1': 'attribute1_value', 'attribute2': 'attribute2_value', ...For example, 'Color': 'White', 'Material': 'Plastic', 'Brand': 'ClearChoice'. Choose one specific value for each attribute.", axis=1)

    for row in df.itertuples():
        id_to_prompt[row.product_id] = row.prompt
        id_to_label[row.product_id] = row.values

    return id_to_prompt, id_to_label


def inference(video_path, prompt, processor, model):
    conversation = [
        {        
            "role": "user",
            "content": [
                {
                    "type": "video", 
                    "video": {"video_path": video_path, "fps": 1, "max_frames": 64}
                },
                {
                    "type": "text", 
                    "text": prompt
                },
            ]
        }
    ]

    # Single-turn conversation
    inputs = processor(conversation=conversation, return_tensors="pt")
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    output_ids = model.generate(**inputs, max_new_tokens=256)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Video inference with VideoLLaMA')
    parser.add_argument('--device_id', type=int, default=5, help='CUDA device ID to use')
    parser.add_argument('--data_path', type=str, default="/Dataset/test_data/clothing_test.csv")
    parser.add_argument('--model_path', type=str, default="DAMO-NLP-SG/VideoLLaMA3-7B", help='Model path')
    parser.add_argument('--num_frames', type=int, default=64, help='Number of frames to use')
    args = parser.parse_args()

    # Get arguments
    model_path = args.model_path
    device_id = args.device_id
    num_frames = args.num_frames
    data_path = args.data_path
    device = f'cuda:{device_id}'  # Changed to string format for device_map

    data_name = data_path.split('/')[-1].split('.')[0]

    output_path_dict = {"appliances_test": "responses_appliances_generalized.csv",
                        "appliances_test_separated": "responses_appliances_w_keys.csv",
                        'arts_test': 'responses_arts_generalized.csv', 
                        'arts_test_separated': 'responses_arts_w_keys.csv',
                        'baby_test': 'responses_baby_generalized.csv',
                        'baby_test_separated': 'responses_baby_w_keys.csv',
                        'grocery_test': 'responses_grocery_generalized.csv',
                        'grocery_test_separated': 'responses_grocery_w_keys.csv',
                        'clothing_test': 'responses_clothing_generalized.csv',
                        'clothing_test_separated': 'responses_clothing_w_keys.csv',
                        'pet_test': 'responses_pet_generalized.csv',
                        'pet_test_separated': 'responses_pet_w_keys.csv',
                        'sports_test': 'responses_sports_generalized.csv',
                        'sports_test_separated': 'responses_sports_w_keys.csv',
                        'musical_test': 'responses_musical_generalized.csv',
                        'musical_test_separated': 'responses_musical_w_keys.csv',
                        'automotive_test': 'responses_automotive_generalized.csv',
                        'automotive_test_separated': 'responses_automotive_w_keys.csv',
                        'toys_test': 'responses_toys_generalized.csv',
                        'toys_test_separated': 'responses_toys_w_keys.csv',
                        'cellphones_test': 'responses_cellphones_generalized.csv',
                        'cellphones_test_separated': 'responses_cellphones_w_keys.csv',
                        'beauty_test': 'responses_beauty_generalized.csv',
                        'beauty_test_separated': 'responses_beauty_w_keys.csv',
                        'patio_test': 'responses_patio_generalized.csv',
                        'patio_test_separated': 'responses_patio_w_keys.csv',
                        'industry_test': 'responses_industry_generalized.csv',
                        'industry_test_separated': 'responses_industry_w_keys.csv',
                        }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device,  # Changed to use specific device
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


    if 'separated' in data_path:
        data = pd.read_csv(data_path)
        # Get prompts and labels
        id_to_prompt, id_to_label = preprocess(data)
        # Process each video
        responses = []
        for url in tqdm(data['content_url']):
            # Get the product_id for this URL
            product_id = data[data['content_url'] == url]['product_id'].iloc[0]
            # Use the specific prompt for this product
            prompt = id_to_prompt[product_id]
            response = inference(url, prompt, processor, model)
            responses.append(response)
            
            
            # Save after each video
            pd.DataFrame(responses, columns=['response']).to_csv(output_path_dict[data_name], index=False)

            # Clean up cache directory
            #shutil.rmtree(f".cache/{data_name}")

        print("Processing completed! Results saved to responses.csv")

    # prompt (generalized)
    else:
        df = pd.read_csv(data_path)
        prompt = "You are a professional product analyst specializing in attribute extraction for leading e-commerce platforms. Your task is to extract the attributes and their values of the product in this video. Answer it in this format only: 'attribute1': 'attribute1_value', 'attribute2': 'attribute2_value', ... For example, 'Color': 'White', 'Material': 'Plastic', 'Brand': 'ClearChoice'. Do not include any kinds of explanation. Choose one specific value for each attribute."
        responses = []

        for url in tqdm(df['content_url']):
            #video_path, frames, timestamps = get_video_frames(url, num_frames=num_frames, cache_dir=f".cache/{data_name}")
            response = inference(url, prompt, processor, model)
            responses.append(response)
                
                # Save after each video
            pd.DataFrame(responses, columns=['response']).to_csv(output_path_dict[data_name], index=False)

            # Clean up cache directory
            #shutil.rmtree(f".cache/{data_name}")
            
        print("Processing completed! Results saved to responses.csv")