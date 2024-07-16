import re
from pathlib import Path

import pandas as pd

MIMIC_III_DIR = Path.cwd() / "physionet.org" / "files" / "mimiciii" / "1.4"

#print(MIMIC_III_DIR)

full_df = pd.read_csv(MIMIC_III_DIR / "NOTEEVENTS.csv")
