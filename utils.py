def json_to_csv(csv_file, json_file, results_dir="results"):
    import json
    import csv
    import os

    data = []
    
    # Read JSON data from file
    json_path = os.path.join(results_dir, f'{json_file}.json')
    with open(json_path, 'r') as infile:
        data = json.load(infile)


    csv_path = os.path.join(results_dir, csv_file)
    with open(csv_path, 'w', newline='') as outfile:
        csv_writer = csv.DictWriter(outfile, fieldnames=data[0].keys())
        
        # Write header
        csv_writer.writeheader()
        
        # Write rows
        csv_writer.writerows(data)

    print(f'Data written to {csv_path}')
