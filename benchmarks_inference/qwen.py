import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import os
import math
import hashlib
import requests
import argparse
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu
import pandas as pd
import shutil

# prompt (provide aspect keys)
def preprocess(df):
    id_to_prompt = {}
    id_to_label = {}

    # Create a prompt that includes the specific attributes for each product
    df['prompt'] = df.apply(lambda row: f"You are a professional product analyst specializing in attribute extraction for leading e-commerce platforms. Your task is to extract the values of the following attributes for the product in this video: {row['attributes']}. Answer it in this format only: 'attribute1': 'attribute1_value', 'attribute2': 'attribute2_value', ... Choose one specific value for each attribute.", axis=1)

    for row in df.itertuples():
        id_to_prompt[row.product_id] = row.prompt
        id_to_label[row.product_id] = row.values

    return id_to_prompt, id_to_label

def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")


def get_video_frames(video_path, num_frames=128, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    if video_path.startswith('http://') or video_path.startswith('https://'):
        video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
        if not os.path.exists(video_file_path):
            download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)

    return video_file_path, frames, timestamps




def inference(video_path, prompt, device, processor, model, max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)


    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Video inference with Qwen2.5-VL')
    parser.add_argument('--device_id', type=int, default=5, help='CUDA device ID to use')
    parser.add_argument('--data_path', type=str, default="/Dataset/test_data/clothing_test.csv")
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help='Model path')
    parser.add_argument('--num_frames', type=int, default=64, help='Number of frames to use')
    args = parser.parse_args()

    # Get arguments
    model_path = args.model_path
    device_id = args.device_id
    num_frames = args.num_frames
    data_path = args.data_path
    device = torch.device(f'cuda:{device_id}')

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
                        'automotive_test': 'responses_automotive_generalized.csv',
                        'automotive_test_separated': 'responses_automotive_w_keys.csv',
                        'beauty_test': 'responses_beauty_generalized.csv',
                        'beauty_test_separated': 'responses_beauty_w_keys.csv',
                        'cellphones_test': 'responses_cellphones_generalized.csv',
                        'cellphones_test_separated': 'responses_cellphones_w_keys.csv',
                        'industry_test': 'responses_industry_generalized.csv',
                        'industry_test_separated': 'responses_industry_w_keys.csv',
                        'musical_test': 'responses_musical_generalized.csv',
                        'musical_test_separated': 'responses_musical_w_keys.csv',
                        'patio_test': 'responses_patio_generalized.csv',
                        'patio_test_separated': 'responses_patio_w_keys.csv',
                        'toys_test': 'responses_toys_generalized.csv',
                        'toys_test_separated': 'responses_toys_w_keys.csv',
                        }
    

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_path)

    if 'separated' in data_path:
        data = pd.read_csv(data_path)
        # Get prompts and labels
        id_to_prompt, id_to_label = preprocess(data)
        # Process each video
        responses = []
        for url in tqdm(data['content_url']):
            video_path, frames, timestamps = get_video_frames(url, num_frames=num_frames, cache_dir=f".cache/{data_name}")
            # Get the product_id for this URL
            product_id = data[data['content_url'] == url]['product_id'].iloc[0]
            # Use the specific prompt for this product
            prompt = id_to_prompt[product_id]
            response = inference(video_path, prompt, device, processor, model)
            responses.append(response)
            
            
            # Save after each video
            pd.DataFrame(responses, columns=['response']).to_csv(output_path_dict[data_name], index=False)

            # Clean up cache directory
            shutil.rmtree(f".cache/{data_name}")

        print("Processing completed! Results saved to responses.csv")

    # prompt (generalized)
    else:
        df = pd.read_csv(data_path)
        prompt = "You are a professional product analyst specializing in attribute extraction for leading e-commerce platforms. Your task is to extract the attributes of the product in this video. Answer it in this format only: 'attribute1': 'attribute1_value', 'attribute2': 'attribute2_value', ... Choose one specific value for each attribute."
        responses = []

        for url in tqdm(df['content_url']):
            video_path, frames, timestamps = get_video_frames(url, num_frames=num_frames, cache_dir=f".cache/{data_name}")
            response = inference(video_path, prompt, device, processor, model)
            responses.append(response)
                
                # Save after each video
            pd.DataFrame(responses, columns=['response']).to_csv(output_path_dict[data_name], index=False)

            # Clean up cache directory
            shutil.rmtree(f".cache/{data_name}")
            
        print("Processing completed! Results saved to responses.csv")
    
    # os.rmdir(f".cache/{data_name}")
    
        
        
        

