import pandas as pd
import matplotlib.pyplot as plt
from sys import argv

if len(argv) != 2:
    print("Usage: python plot_scores_csv.py <csv_file>")
    sys.exit(1)


df = pd.DataFrame(columns=["Levenshtein distance", "BLEU score", "METEOR score"])

df = pd.read_csv(argv[1])

ax = df["BLEU score"].plot()
ax.set_xlabel("Sample")
ax.set_ylabel("BLEU")
plt.show()
