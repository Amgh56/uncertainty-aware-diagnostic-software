import pandas as pd 

df = pd.read_csv("NIH_dataset/calibration_2500.csv")

print([df["No Finding"].sum()])
