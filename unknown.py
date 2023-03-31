import csv

# Open the first CSV file in append mode
with open('database.csv', mode='a', newline='') as file1:
    writer = csv.writer(file1)

    # Open the second CSV file in read mode
    with open('unknown.csv', mode='r') as file2:
        reader = csv.reader(file2)

        # Skip the header row of the second CSV file
        next(reader)

        # Append the remaining rows to the first CSV file
        for row in reader:
            writer.writerow(row)

print('Data appended successfully!')
