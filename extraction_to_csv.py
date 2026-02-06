import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('json_file', type=str, help='Path to the JSON file')
args = parser.parse_args()

data = json.load(open(args.json_file))

extracted_data = []
for material in data['materials']:
    extracted_data.append({
        'material_name': material.get('canonical_formula') if material.get('canonical_formula') else material['electrolyte_name']['full_name'],
        'material_class': material.get('material_class'),
        'measurement_temperature': material.get('measurement_temperature'),
        'ionic_conductivity': material.get('ionic_conductivity_S_per_cm'),
        'material_description': material.get('material_description'),
        'processing_method': material.get('processing_method')
    })

# Create DataFrame
df = pd.DataFrame(extracted_data)

# Save to CSV
csv_filename = args.json_file.replace('.json', '.csv')
df.to_csv(csv_filename, index=False)

# Display the first few rows
print(df[['material_name', 'ionic_conductivity']])