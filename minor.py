import pandas as pd

df = pd.read_csv("export_transfer.csv")
# print(df)
# print(df.columns)
df = df[["To", "Signature", "Human Time"]].sort_values(by="Human Time", ascending=True)
df = df.groupby("To").last().reset_index()
d = df[["To", "Signature"]].set_index("To").to_dict()["Signature"]
print(d)