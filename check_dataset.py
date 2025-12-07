import os
import pandas as pd

# Load CSV
trainLabels = pd.read_csv("trainLabels.csv")

# Normalize for safe comparison
trainLabels['image'] = trainLabels['image'].astype(str).str.strip().str.lower()

# Get dataset filenames without extensions
listing = os.listdir("Dataset/")
listing = [os.path.splitext(f)[0].lower() for f in listing]

print("Total images in Dataset/:", len(listing))
print("Total entries in CSV:", len(trainLabels))

# Convert to sets
dataset_set = set(listing)
csv_set = set(trainLabels['image'])

# Find mismatches
missing_in_csv = dataset_set - csv_set
missing_in_dataset = csv_set - dataset_set

if missing_in_csv:
    print(f"\n⚠️ {len(missing_in_csv)} files are in Dataset/ but NOT in CSV.")
    print("Examples:", list(missing_in_csv)[:20])

if missing_in_dataset:
    print(f"\n⚠️ {len(missing_in_dataset)} entries are in CSV but NOT in Dataset/.")
    print("Examples:", list(missing_in_dataset)[:20])

if not missing_in_csv and not missing_in_dataset:
    print("\n✅ All dataset files and CSV entries match perfectly!")
