import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
import requests
from urllib.parse import urlparse
import tempfile
import argparse
import pandas as pd
from tqdm import tqdm
import shutil


# model setting
model_path = 'OpenGVLab/InternVL_2_5_HiCo_R16'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices

def get_num_frames_by_duration(duration):
        local_num_frames = 4        
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments
        
        num_frames = min(512, num_frames)
        num_frames = max(64, num_frames)

        return num_frames

def download_video(url):
    """Download video from URL to a temporary file."""
    try:
        print(f"Attempting to download video from: {url}")
        
        # Create a temporary file with .mp4 extension
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_path = temp_file.name
        temp_file.close()
        print(f"Created temporary file at: {temp_path}")

        # Download the video with timeout and headers
        '''
        headers = {
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        '''
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        print(f"Total file size: {total_size} bytes")

        # Write the video content to the temporary file
        with open(temp_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Print progress
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"Download progress: {progress:.1f}%", end='\r')

        print("\nDownload completed successfully")
        return temp_path
    except requests.exceptions.RequestException as e:
        print(f"Network error while downloading: {e}")
        raise
    except Exception as e:
        print(f"Error downloading video: {e}")
        raise

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, get_frame_by_duration=False):
    """Load video from path or URL with enhanced error handling."""
    try:
        # Check if the path is a URL
        if video_path.startswith(('http://', 'https://')):
            print(f"Processing video from URL: {video_path}")
            # Download the video first
            local_path = download_video(video_path)
            try:
                print(f"Attempting to read video from local path: {local_path}")
                # Verify file exists and has content
                if not os.path.exists(local_path):
                    raise FileNotFoundError(f"Downloaded file not found at {local_path}")
                if os.path.getsize(local_path) == 0:
                    raise ValueError("Downloaded file is empty")

                # Use the local path for video reading
                vr = VideoReader(local_path, ctx=cpu(0), num_threads=1)
                print(f"Successfully loaded video with {len(vr)} frames")
                
                max_frame = len(vr) - 1
                fps = float(vr.get_avg_fps())
                print(f"Video FPS: {fps}")

                pixel_values_list, num_patches_list = [], []
                transform = build_transform(input_size=input_size)
                if get_frame_by_duration:
                    duration = max_frame / fps
                    num_segments = get_num_frames_by_duration(duration)
                frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
                print(f"Processing {len(frame_indices)} frames")
                
                for frame_index in frame_indices:
                    img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
                    img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
                    pixel_values = [transform(tile) for tile in img]
                    pixel_values = torch.stack(pixel_values)
                    num_patches_list.append(pixel_values.shape[0])
                    pixel_values_list.append(pixel_values)
                
                pixel_values = torch.cat(pixel_values_list)
                print("Video processing completed successfully")
                return pixel_values, num_patches_list
            finally:
                # Clean up the temporary file
                if os.path.exists(local_path):
                    os.unlink(local_path)
                    print(f"Cleaned up temporary file: {local_path}")
        else:
            # Use the original path if it's a local file
            print(f"Processing local video file: {video_path}")
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found at {video_path}")
                
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            max_frame = len(vr) - 1
            fps = float(vr.get_avg_fps())

            pixel_values_list, num_patches_list = [], []
            transform = build_transform(input_size=input_size)
            if get_frame_by_duration:
                duration = max_frame / fps
                num_segments = get_num_frames_by_duration(duration)
            frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
            for frame_index in frame_indices:
                img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
                img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
                pixel_values = [transform(tile) for tile in img]
                pixel_values = torch.stack(pixel_values)
                num_patches_list.append(pixel_values.shape[0])
                pixel_values_list.append(pixel_values)
            pixel_values = torch.cat(pixel_values_list)
            return pixel_values, num_patches_list

    except Exception as e:
        print(f"Error loading video: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        raise

# evaluation setting
max_num_frames = 512
generation_config = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.1,
    num_beams=1
)
num_segments=128


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video inference with InternVL-2.5')
    parser.add_argument('--device_id', type=int, default=5, help='CUDA device ID to use')
    parser.add_argument('--data_path', type=str, default="/Dataset/test_data/clothing_test.csv")
    parser.add_argument('--model_path', type=str, default="OpenGVLab/InternVL_2_5_HiCo_R16", help='Model path')
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

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to(device)

    # Generation config
    generation_config = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=1024,
        top_p=0.1,
        num_beams=1
    )

    if 'separated' in data_path:
        data = pd.read_csv(data_path)
        id_to_prompt, id_to_label = preprocess(data)
        # Process each video
        responses = []
        for url in tqdm(data['content_url']):
            try:
                with torch.no_grad():
                    pixel_values, num_patches_list = load_video(url, num_segments=num_frames, max_num=1, get_frame_by_duration=False)
                    pixel_values = pixel_values.to(torch.float16).to(device)
                    video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
                    
                    # Get the product_id for this URL
                    product_id = data[data['content_url'] == url]['product_id'].iloc[0]
                    # Get the specific attributes for this product
                    #attributes = data[data['content_url'] == url]['attributes'].iloc[0]
                    question = id_to_prompt[product_id]
                    #question = f"You are a professional product analyst specializing in attribute extraction for leading e-commerce platforms. Your task is to extract the values of the following attributes for the product in this video: {attributes}. Answer it in this format only: 'attribute1': 'attribute1_value', 'attribute2': 'attribute2_value', ... Choose one specific value for each attribute."
                    
                    # Combine video prefix and question
                    full_prompt = video_prefix + question
                    output, _ = model.chat(tokenizer, pixel_values, full_prompt, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
                    responses.append(output)
            except Exception as e:
                print(f"Error processing video {url}: {str(e)}")
                responses.append("Error: " + str(e))

            # Save after each video
            pd.DataFrame(responses, columns=['response']).to_csv(output_path_dict[data_name], index=False)

        print(f"Processing completed! Results saved to {output_path_dict[data_name]}")

    else:
        df = pd.read_csv(data_path)
        responses = []
        question = "You are a professional product analyst specializing in attribute extraction for leading e-commerce platforms. Your task is to extract the attributes and their values of the product in this video. Answer it in this format only: 'attribute1': 'attribute1_value', 'attribute2': 'attribute2_value', ... For example, 'Color': 'White', 'Material': 'Plastic', 'Brand': 'ClearChoice'. Do not include any kinds of explanation. Choose one specific value for each attribute."
        #question = "You are a professional product analyst specializing in attribute extraction for leading e-commerce platforms. Your task is to extract the attributes and their values of the product in this video. Answer it in this format only: 'attribute1': 'attribute1_value', 'attribute2': 'attribute2_value', ... Choose one specific value for each attribute."
        for url in tqdm(df['content_url']):
            try:
                with torch.no_grad():
                    pixel_values, num_patches_list = load_video(url, num_segments=num_frames, max_num=1, get_frame_by_duration=False)
                    pixel_values = pixel_values.to(torch.float16).to(device)
                    video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
                    
                    # Combine video prefix and question
                    full_prompt = video_prefix + question
                    output, _ = model.chat(tokenizer, pixel_values, full_prompt, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
                    responses.append(output)
            except Exception as e:
                print(f"Error processing video {url}: {str(e)}")
                responses.append("Error: " + str(e))

            # Save after each video
            pd.DataFrame(responses, columns=['response']).to_csv(output_path_dict[data_name], index=False)

        print(f"Processing completed! Results saved to {output_path_dict[data_name]}")