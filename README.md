# VideoAVE

An open-source video-to-text attribute value extraction dataset.

## Dataset

* 224k training data, 25k testing data, 14 domains, 172 unique attributes

## Getting Started

### Data Pre-processing

For processing from raw data:
```
cd data_processing
python raw_data_processing.py --data_name raw_meta_All_Beauty --output_path raw_meta_All_Beauty.csv
```
For implementing CLIP-MoE data curation steps:
```
python CLIP-MoE.py
```
The output will be the same as training data and testing data provided in Dataset folder. 

### Benchmarks Inference

We support four benchmarks: Video-LLaVA, VideoLLaMA3, InternVideo2.5, and Qwen2.5-VL. 

Here is an example for Qwen2.5-VL:
```
cd benchmarks_inference
python qwen.py
```

### Benchmarks Finetuning

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for fine-tuning Qwen2.5-VL. data_info and yaml files are provided in model_training. Here is the fine-tuning steps:

* Clone [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory) Repo and setup the environment.
* Generating training data in LLaMA-Factory data format. Please refer to:
```
 ./data_processing/training_data_processing.ipynb
```
* Update yaml file as:
```
 ./model_training/qwen2_5vl_full_sft.yaml
```
* Start training

### Evaluation

```
 evaluation.ipynb
```
