import os

for i in range(1,7):
    str = """Arbeitsargentur fuer Arbeit: Jobs zuletzt bearbeitet
https://www.arbeitsagentur.de/jobsuche/suche?angebotsart=1&sort=moddatum"""
    with open(f"data/jobs/10/processed_data/100000{i}.txt", "w") as f:
        f.write(str)