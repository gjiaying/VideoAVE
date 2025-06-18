from datasets import load_dataset
import json
import pandas as pd
import os
import argparse

category = ['raw_meta_All_Beauty', 'raw_meta_Toys_and_Games', 'raw_meta_Cell_Phones_and_Accessories', 'raw_meta_Industrial_and_Scientific', 'raw_meta_Gift_Cards', 'raw_meta_Musical_Instruments', 'raw_meta_Electronics', 'raw_meta_Handmade_Products', 'raw_meta_Arts_Crafts_and_Sewing', 'raw_meta_Baby_Products', 'raw_meta_Health_and_Household', 'raw_meta_Office_Products', 'raw_meta_Digital_Music', 'raw_meta_Grocery_and_Gourmet_Food', 'raw_meta_Sports_and_Outdoors', 'raw_meta_Home_and_Kitchen', 'raw_meta_Subscription_Boxes', 'raw_meta_Tools_and_Home_Improvement', 'raw_meta_Pet_Supplies', 'raw_meta_Video_Games', 'raw_meta_Kindle_Store', 'raw_meta_Clothing_Shoes_and_Jewelry', 'raw_meta_Patio_Lawn_and_Garden', 'raw_meta_Unknown', 'raw_meta_Books', 'raw_meta_Automotive', 'raw_meta_CDs_and_Vinyl', 'raw_meta_Beauty_and_Personal_Care', 'raw_meta_Amazon_Fashion', 'raw_meta_Magazine_Subscriptions', 'raw_meta_Software', 'raw_meta_Health_and_Personal_Care', 'raw_meta_Appliances', 'raw_meta_Movies_and_TV']

def raw_data_loading(raw_data):
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", raw_data, trust_remote_code=True, cache_dir="/data/datasets")
    return dataset

def dataset_filtering(dataset, desired_keys, output_file):
    filtered_data = []
    for i in range(len(dataset['full'])):
        record = dataset['full'][i]
        if len(record.get('videos').get('url')) != 0:
            # Load the 'details' field as a dictionary
            details = json.loads(record.get('details', '{}'))  # Note the fallback to '{}' to avoid TypeError
            
            # Filter 'details' to keep only desired keys
            filtered_details = {key: value for key, value in details.items() if key in desired_keys}
            
            # Keep only if there are more than one attribute-value pairs
            if len(filtered_details) > 1:
                filtered_record = {
                    'parent_asin': record.get('parent_asin'),
                    'videos': record.get('videos'),
                    'title': record.get('title'),
                    'details': filtered_details
                }
                filtered_data.append(filtered_record)

    # Create DataFrame
    df = pd.DataFrame(filtered_data)

    # Save the DataFrame to a CSV file
    csv_file_path = 'outputfile.csv'
    df.to_csv(csv_file_path, index=False)

    print(f"DataFrame saved to {csv_file_path}")


def main():
    parser = argparse.ArgumentParser(description='Raw Dataset Processing.')
    parser.add_argument('--data_name', type=str, default="raw_meta_All_Beauty")
    parser.add_argument('--output_path', type=str, default="raw_meta_All_Beauty.csv")
    parser.add_argument('--desired_keys', type=json, default={'Brand', 'Color', 'Material'})
    args = parser.parse_args()
    data_name = args.data_name
    desired_keys = args.desired_keys
    output_path = args.output_path

    dataset = raw_data_loading(data_name)
    dataset_filtering(dataset, desired_keys, output_path)




if __name__ == "__main__":
    main()