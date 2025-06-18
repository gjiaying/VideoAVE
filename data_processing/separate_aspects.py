import pandas as pd
import ast

# Read the CSV file
df = pd.read_csv('/Dataset/test_data/clothing_test.csv')

# Convert string representation of dictionary to actual dictionary
def parse_aspects(aspects_str):
    try:
        # Remove single quotes and convert to double quotes for proper JSON parsing
        aspects_str = aspects_str.replace("'", '"')
        return ast.literal_eval(aspects_str)
    except:
        return {}

# Apply the parsing function to the aspects column
df['aspects_dict'] = df['aspects'].apply(parse_aspects)

# Create lists to store attributes and values
all_attributes = []
all_values = []

# Extract attributes and values from each row
for aspects in df['aspects_dict']:
    attributes = list(aspects.keys())
    values = list(aspects.values())
    
    # Convert lists to strings with comma separator
    attributes_str = ','.join(attributes)
    values_str = ','.join(values)
    
    all_attributes.append(attributes_str)
    all_values.append(values_str)

# Add new columns to the dataframe
df['attributes'] = all_attributes
df['values'] = all_values

# Drop the temporary dictionary column
df = df.drop('aspects_dict', axis=1)

# Save the processed data to a new CSV file
df.to_csv('/Dataset/test_data_separated/clothing_test_separated.csv', index=False) 