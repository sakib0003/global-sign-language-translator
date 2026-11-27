
import pandas as pd

TRAIN = "datasets/processed/bdsl126_pose_landmarks_train.csv"
VAL = "datasets/processed/bdsl126_pose_landmarks_val.csv"

train = pd.read_csv(TRAIN)
val = pd.read_csv(VAL)

def summarize(df):
    frame = df["source"].str.contains(
        r"frame\d+", case=False, regex=True
    )
    result = pd.DataFrame({
        "label": df["label"],
        "type": frame.map({
            True: "Frame",
            False: "Individual"
        })
    })

    return result.pivot_table(
        index="label",
        columns="type",
        aggfunc="size",
        fill_value=0
    )

a = summarize(train)
b = summarize(val)

report = a.add_prefix("Train_").join(
    b.add_prefix("Val_"), how="outer"
).fillna(0).astype(int)

for col in ["Train_Frame", "Train_Individual",
            "Val_Frame", "Val_Individual"]:
    if col not in report:
        report[col] = 0

report = report.sort_index()

print("\n===== SAMPLE TYPE COUNTS =====")
print(report.to_string())

print("\n===== TOTALS =====")
print(report.sum().to_string())

print("\n===== CLASSES WITHOUT INDIVIDUAL IMAGES =====")
print(report[
    (report["Train_Individual"] + report["Val_Individual"]) == 0
].index.tolist())

report.to_csv(
    "reports/bdsl126_sample_type_audit.csv"
)
print("\nSaved: reports/bdsl126_sample_type_audit.csv")
