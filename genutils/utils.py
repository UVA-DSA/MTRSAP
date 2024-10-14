import json
import csv

data = []


def json_to_csv(csv_file, json_file):
    
    # Read JSON data from file
    with open(f'results/{json_file}.json', 'r') as json_file:
        data = json.load(json_file)


    csv_path = f'results/{csv_file}'
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())
        
        # Write header
        csv_writer.writeheader()
        
        # Write rows
        csv_writer.writerows(data)

    print(f'Data written to {csv_path}')
    

