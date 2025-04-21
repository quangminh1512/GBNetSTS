import pandas as pd

input_file = './sts-dev.csv'  # Adjust path if needed
output_file = './output_2017_data.xlsx'

data = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 7 and parts[2] == '2017':
            try:
                score = float(parts[4]) / 5  # Normalize to 0–1
                sent1 = parts[5]
                sent2 = parts[6]
                data.append([sent1, sent2, score])
            except ValueError:
                continue  # Skip lines with invalid score values

# Create DataFrame
df = pd.DataFrame(data, columns=['Câu 1', 'Câu 2', 'Similarity Score'])

# Save to Excel
df.to_excel(output_file, index=False)

print(f"Excel file saved as {output_file}")