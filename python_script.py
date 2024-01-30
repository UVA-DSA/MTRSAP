original_list = [
    [0, 7, 6, 6],
    [1, 7, 6, 6],
    [2, 7, 8, 8],
    [3, 7, 8, 8]
]

expanded_list = []

for i, row in enumerate(original_list):
    for j in range(65):  # Count from 0 to 64
        expanded_row = [j] + row[1:]  # Change the first element
        expanded_list.append(expanded_row)
    
print(expanded_list)