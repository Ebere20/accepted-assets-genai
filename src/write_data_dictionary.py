"""Write data/data_dictionary.csv: one row per column of the simulated dataset."""
import ast, csv, pathlib
import pandas as pd

src = pathlib.Path(__file__).with_name("build_workbook.py").read_text()
start = src.index("DICT = {"); end = src.index("\n}", start) + 2
DICT = ast.literal_eval(src[start + len("DICT = "):end])
cols = pd.read_csv("data/simulated_tasks_50000.csv", nrows=5).columns
with open("data/data_dictionary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["variable", "description"])
    for c in cols:
        w.writerow([c, DICT.get(c, "")])
print("data/data_dictionary.csv:", len(cols), "variables,", sum(c in DICT for c in cols), "described")
