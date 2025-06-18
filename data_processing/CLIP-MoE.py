import av
import torch
import pandas as pd
import numpy as np
import os
import json
import ast
from scipy.stats import zscore
from collections import Counter
import requests
import argparse
from argparse import Namespace
from IPython.display import HTML, display
#import matplotlib.pyplot as plt
import video_clip
from simple_tokenizer import SimpleTokenizer as _Tokenizer
from viclip import ViCLIP
import cv2

from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download

np.random.seed(0)
v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
clip_candidates = {'viclip':None, 'clip':None}


# Function to clean and filter the 'details' column
def filter_details(details):
    try:
        # Convert the string to a dictionary (this assumes it's a string representation of a dictionary)
        details_dict = ast.literal_eval(details)
    except:
        # In case there's any parsing error, return the original details
        return details

    # Filter out attributes where value contains a comma or more than 3 words
    filtered_details = {k: v for k, v in details_dict.items() if len(v.split()) <= 3 and ',' not in v}

    # If only one attribute-value pair remains, remove this record
    if len(filtered_details) == 1:
        return None  # Return None to indicate we should drop this record

    # Return the filtered details as a string
    return str(filtered_details)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
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


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def get_clip(name='viclip'):
    global clip_candidates
    m = clip_candidates[name]
    if m is None:
        if name == 'viclip':
            tokenizer = _Tokenizer()
            vclip = ViCLIP(tokenizer)
            # m = vclip
            m = (vclip, tokenizer)
        else:
            raise Exception('the target clip model is not found.')
    
    return m

def get_text_feat_dict(texts, clip, tokenizer, text_feat_d={}):
    for t in texts:
        feat = clip.get_text_features(t, tokenizer, text_feat_d)
        text_feat_d[t] = feat
    return text_feat_d

def get_vid_feat(frames, clip):
    return clip.get_vid_features(frames)

def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

def normalize(data):
    return (data/255.0-v_mean)/v_std

def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert(len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube

def retrieve_text(frames, texts, clip, tokenizer, name='viclip', topk=5, device=torch.device('cuda')):
    #clip, tokenizer = get_clip(name)
    #clip = clip.to(device)
    frames_tensor = frames2tensor(frames, device=device)
    vid_feat = get_vid_feat(frames_tensor, clip)

    text_feat_d = {}
    text_feat_d = get_text_feat_dict(texts, clip, tokenizer, text_feat_d)
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, 0)
    
    #probs, idxs = clip.get_predict_label(vid_feat, text_feats_tensor, top=topk)
    score = vid_feat @ text_feats_tensor.T


    #ret_texts = [texts[i] for i in idxs.numpy()[0].tolist()]
    return score.item()

def download_mp4(url, save_path):
    """
    Download an MP4 video from a URL and save it to the given local path.
    
    Args:
        url (str): The URL of the .mp4 video.
        save_path (str): Local file path where the video should be saved (including .mp4).
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        #print(f"Downloaded to {save_path}")
    else:
        print(f"Failed to download video. Status code: {response.status_code}")


def xclip_MoE(data):
    processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
    model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    xclip_scores = []
    videos = data['content_url'].tolist()
    titles = data['parent_title'].tolist()

    for i in range(len(titles)):  # or use `range(len(data))` for full dataset
        try:
            container = av.open(videos[i])
            indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            video = read_video_pyav(container, indices)

            inputs = processor(
                text=[titles[i]],
                videos=list(video),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            logits_per_video = outputs.logits_per_video
            score = logits_per_video.item()  # assuming 1 video-text pair, so scalar
            xclip_scores.append(score)

        except Exception as e:
            print(f"Error processing index {i}: {e}")
            xclip_scores.append(None)

    # Add new column and save
    #data['x-clip_score'] = xclip_scores
    #data.to_csv('./mp4_data/beauty_with_xclip_scores.csv', index=False)
    return xclip_scores

def videoclip_MoE(data):
    eval_config = 'eval_configs/video_clip_v0.3.yaml'
    model, vis_processor = video_clip.load_model(eval_config)
    videos = data['content_url'].tolist()
    titles = data['parent_title'].tolist()
    videoclip_scores = []

    for i in range(len(titles)):
        video_path = f'./data/{i}.mp4'
        try:
            download_mp4(videos[i], video_path)
            texts = [titles[i]]
            video_embs = video_clip.get_all_video_embeddings([video_path], model, vis_processor)
            sims = video_clip.compute_sim(model, texts, video_embs)
            sims = torch.cat(sims).detach().cpu().numpy()
            videoclip_scores.append(sims[0][0])
        except:
            videoclip_scores.append(0)
            continue

        if os.path.exists(video_path):
            os.remove(video_path)
    # Add new column and save
    #data['videoclip_score'] = videoclip_scores
    #data.to_csv('../mp4_data/beauty_with_videoclip_scores.csv', index=False)
    return videoclip_scores

def viclip_MoE(data):
    device = torch.device('cuda')
    videos = data['content_url'].tolist()
    titles = data['parent_title'].tolist()
    viclip_scores = []
    clip, tokenizer = get_clip(name='viclip')
    clip = clip.to(device)
    for i in range(len(titles)):
        video_path = f'./data/{i}.mp4'
        try:
            download_mp4(videos[i], video_path)
            video = cv2.VideoCapture(video_path)
            frames = [x for x in _frame_from_video(video)]
            text_candidates = [titles[i]]
            score = retrieve_text(frames, text_candidates, clip, tokenizer, name='viclip', topk=1)
            viclip_scores.append(score)
            torch.cuda.empty_cache()
        except:
            viclip_scores.append(0)
            continue


        if os.path.exists(video_path):
            os.remove(video_path)
        del frames, text_candidates, score
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    #data['viclip_score'] = viclip_scores
    #data.to_csv('../mp4_data/beauty_with_viclip_scores.csv', index=False)  
    return viclip_scores  

def MoE_filtering(df):
    score_cols = df.columns[-3:]
    score_data = df[score_cols]

    # Compute z-scores for each column
    z_scores = score_data.apply(zscore)

    # Apply filtering: Keep rows where at least 2 z-scores are above a threshold (e.g., -1.0)
    threshold = -1.0
    mask = (z_scores > threshold).sum(axis=1) >= 2
    filtered_df = df[mask]

    # === Save to new CSV ===
    #filtered_df.to_csv('./final_data/sports.csv', index=False)

    #print(f"Filtered dataset saved with {len(filtered_df)} samples.")
    return filtered_df

def main():
    parser = argparse.ArgumentParser(description='Raw Dataset Processing.')
    parser.add_argument('--data_path', type=str, default="/Dataset/data_mp4/raw_meta_Beauty_and_Personal_Care_video_content_urls.csv")
    parser.add_argument('--train_path', type=str, default="/Dataset/train_data/beauty_train.csv")
    parser.add_argument('--test_path', type=str, default="/Dataset/test_data/beauty_test.csv")
    args = parser.parse_args()
    data_path = args.data_path
    train_path = args.train_path
    test_path = args.test_path
    data = pd.read_csv(data_path)
    
    xclip_scores = xclip_MoE(data)
    videoclip_scores = videoclip_MoE(data)
    viclip_scores = viclip_MoE(data)
    data['x-clip_score'] = xclip_scores
    data['videoclip_score'] = videoclip_scores
    data['viclip_score'] = viclip_scores
    

    df = MoE_filtering(data)
    # further cleaning
    df['aspects'] = df['details'].apply(filter_details)
    # Remove rows where filtered details are None (i.e., only one attribute-value pair)
    df = df[df['aspects'].notna()]
    # Optionally, remove the old 'details' column if you don't need it anymore
    df.drop('details', axis=1, inplace=True)
    df.drop(['x-clip_score', 'videoclip_score', 'viclip_score'], axis=1, inplace=True)

    #Train-Eval splitting
    # Randomly select 10% of the data
    sampled_df = df.sample(frac=0.1, random_state=42)  # random_state ensures reproducibility
    sampled_df.to_csv(test_path, index=False)
    remaining_df = df.drop(sampled_df.index)
    remaining_df.to_csv(train_path, index=False)
    print('Finish Data Pre-processing.')




if __name__ == "__main__":
    main()