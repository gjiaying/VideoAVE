import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration, BitsAndBytesConfig
import av
import numpy as np
from huggingface_hub import hf_hub_download

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "LanguageBind/Video-LLaVA-7B-hf"


def preprocess(df):
    id_to_prompt = {}
    id_to_label = {}

    # Create a prompt that includes the specific attributes for each product
    #df['prompt'] = df.apply(lambda row: f"You are a professional product analyst specializing in attribute extraction for leading e-commerce platforms. Your task is to extract the values of the following attributes for the product in this video: {row['attributes']}. Answer it in this format only: 'attribute1': 'attribute1_value', 'attribute2': 'attribute2_value', ... Choose one specific value for each attribute.", axis=1)
    df['prompt'] = df.apply(lambda row: f"Your task is to extract only the values for the following attributes for the product in this video: {row['attributes']}. Do not infer or add any other attributes. Respond only in this format: 'attribute1': 'value1', 'attribute2': 'value2', ...For example, 'Color': 'White', 'Material': 'Plastic', 'Brand': 'ClearChoice'. Choose one specific value for each attribute. No explanation or extra text.", axis=1)
    for row in df.itertuples():
        id_to_prompt[row.product_id] = row.prompt
        id_to_label[row.product_id] = row.values

    return id_to_prompt, id_to_label

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.
    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def inference(video_path, prompt, processor, model):
    """
    Run inference on a video using the Video-LLaVA model.
    
    Args:
        video_path (str): Path to the video file
        prompt (str): The prompt to use for generation
        processor: The video processor
        model: The Video-LLaVA model
        
    Returns:
        str: Generated text response (only the assistant's answer)
    """
    # Open video and sample frames
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    clip = read_video_pyav(container, indices)
    
    # Format prompt
    if not prompt.startswith("USER: "):
        prompt = f"USER: <video>\n{prompt} ASSISTANT:"
    
    # Process inputs
    inputs = processor(prompt, videos=clip, return_tensors="pt").to(model.device)
    
    # Generate response
    generate_kwargs = {
        "max_new_tokens": 100,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 2
    }
    
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]
    
   
    answer = generated_text.split("ASSISTANT:")[-1].strip()
        
    return answer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Video inference with Video-LLaVA')
    parser.add_argument('--device_id', type=int, default=5, help='CUDA device ID to use')
    parser.add_argument('--data_path', type=str, default="/Dataset/test_data/clothing_test.csv")
    parser.add_argument('--model_path', type=str, default="LanguageBind/Video-LLaVA-7B", help='Model path')
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
    
    model_id = "LanguageBind/Video-LLaVA-7B-hf"

    processor = VideoLlavaProcessor.from_pretrained(model_id)
    model = VideoLlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config).to(device)
    

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

        print("Processing completed! Results saved to responses.csv")

    # prompt (generalized)
    else:
        df = pd.read_csv(data_path)
        prompt = "Extract attributes and their corresponding values from the product in this video. Use real attribute names and assign only one clear, specific value to each. Respond only in this format (no extra text): 'attribute1': 'value1', 'attribute2': 'value2', ...For example, 'Color': 'Red', 'Material': 'Plastic', 'Brand': 'Nike'. Do not make up attributes, use generic placeholders, or include any explanations."
        # prompt = "Your task is to extract attributes and their corresponding values of the product in this video. Format your output exactly as: 'Color': 'Red', 'Material': 'Plastic', 'Brand': 'Nike' — using real attribute names and one clear value per attribute. Do not use placeholders like 'attribute1' or 'value1'. Do not include any explanations. Choose one specific value for each attribute."
        responses = []

        for url in tqdm(df['content_url']):
            response = inference(url, prompt, processor, model)
            responses.append(response)
                
                # Save after each video
            pd.DataFrame(responses, columns=['response']).to_csv(output_path_dict[data_name], index=False)

            # Clean up cache directory
            
        print("Processing completed! Results saved to responses.csv")
    
    # os.rmdir(f".cache/{data_name}")