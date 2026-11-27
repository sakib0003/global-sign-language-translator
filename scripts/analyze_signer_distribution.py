
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(
    ROOT,
    "datasets",
    "processed",
    "bdsl47_landmarks.csv"
)

OUTPUT_DIR = os.path.join(ROOT, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(CSV_PATH)

# Alphabet dataset only
alphabet = df[
    df["label"].astype(str).str.contains(
        "Sign Alphabets",
        case=False
    )
].copy()

if alphabet.empty:
    raise ValueError("Alphabet dataset not found.")

required_columns = ["label", "signer"]

for col in required_columns:
    if col not in alphabet.columns:
        raise ValueError(f"Missing column: {col}")

# Extract class name
alphabet["class"] = alphabet["label"].astype(str).str.split("__").str[-1]

print("\nTotal Alphabet Samples:", len(alphabet))
print("Total Classes:", alphabet["class"].nunique())
print("Total Signers:", alphabet["signer"].nunique())

# Samples per class
class_counts = alphabet.groupby("class").size()

# Samples per signer
signer_counts = alphabet.groupby("signer").size()

# Class x signer distribution
distribution = pd.crosstab(
    alphabet["class"],
    alphabet["signer"]
)

print("\nSamples Per Class:")
print(class_counts.to_string())

print("\nSamples Per Signer:")
print(signer_counts.to_string())

print("\nClass x Signer Distribution:")
print(distribution.to_string())

# Save reports
class_counts.to_csv(
    os.path.join(OUTPUT_DIR, "alphabet_class_counts.csv")
)

signer_counts.to_csv(
    os.path.join(OUTPUT_DIR, "alphabet_signer_counts.csv")
)

distribution.to_csv(
    os.path.join(OUTPUT_DIR, "alphabet_class_signer_distribution.csv")
)

# Check unusual class counts
print("\nClasses with unusually low sample counts:")

low_classes = class_counts[class_counts < class_counts.median() * 0.8]

if low_classes.empty:
    print("No unusually low-count classes found.")
else:
    print(low_classes.to_string())

# Plot class x signer heatmap
plt.figure(figsize=(16, 12))

plt.imshow(distribution.values, aspect="auto")
plt.colorbar(label="Number of Samples")

plt.xticks(
    range(len(distribution.columns)),
    distribution.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(distribution.index)),
    distribution.index
)

plt.title("BdSL Alphabet - Class and Signer Distribution")
plt.xlabel("Signer")
plt.ylabel("Class")

plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "alphabet_signer_distribution.png"
)

plt.savefig(plot_path, dpi=180)
plt.show()

print("\nAnalysis completed!")
print("Reports saved in:", OUTPUT_DIR)
