# ProjectAnimePopAnalysis

# Anime Popularity Trend Analysis
# Dataset: popular_anime.csv (from MyAnimeList)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# read in the CSV
# ------------------------------------------------------------------------------------------

df = pd.read_csv("project/popular_anime.csv")

print("Dataset loaded!")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}\n")

# STEP 2: Clean the data
# Drop rows where score or rank is missing since those are the most important
# Also reset the index so it's clean after dropping
# ------------------------------------------------------------------------------------------

df = df.dropna(subset=["score", "rank", "scored_by"])
df = df.reset_index(drop=True)

# Convert score and episodes to numbers so we can do math on them
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
df["scored_by"] = pd.to_numeric(df["scored_by"], errors="coerce")

print(f"Rows after cleaning: {len(df)}\n")

# -----------------------------------------------

print("Basic Score Statistics")
print(df["score"].describe())
print()

# What types of anime are in here?
print("Anime Types in Dataset")
print(df["type"].value_counts())
print()


#  Genre Analysis
# Each anime can have multiple genres separated by commas
# split anime with multiple genres seperated by commas to track which genres tend to score higher
# -----------------------------------------------

genre_scores = defaultdict(list)

for _, row in df.iterrows():
    if pd.notna(row["genres"]):
        genres = row["genres"].split(",")
        for genre in genres:
            genre = genre.strip()
            if genre:
                genre_scores[genre].append(row["score"])

# Calculate the average score per genre
# Only keep genres that have at least 50 anime otherwise the average isn't reliable
genre_avg = {}
for genre, scores in genre_scores.items():
    if len(scores) >= 50:
        genre_avg[genre] = round(sum(scores) / len(scores), 2)

# Sort from highest to lowest
genre_avg_sorted = dict(sorted(genre_avg.items(), key=lambda x: x[1], reverse=True))

print("=== Average Score by Genre (top 10) ===")
for genre, avg in list(genre_avg_sorted.items())[:10]:
    print(f"  {genre}: {avg}")
print()


# type Analysis (TV, Movie, OVA, etc.)
# Which format is betterrrr
# -----------------------------------------------------------------------

type_avg = df.groupby("type")["score"].mean().sort_values(ascending=False)

print("=== Average Score by Type ===")
print(type_avg.round(2))
print()

# Episode Count vs Score
# group episode counts into bucketsto compare
# --------------------------------------------------------------------------

# Create episode bins using pd.cut
bins = [0, 12, 24, 50, 100, 10000]
labels = ["1-12", "13-24", "25-50", "51-100", "100+"]

df["episode_range"] = pd.cut(df["episodes"], bins=bins, labels=labels)
ep_avg = df.groupby("episode_range", observed=True)["score"].mean().round(2)

print("=== Average Score by Episode Count ===")
print(ep_avg)
print()

# Age Rating vs Score

# -------------------------------------------------------------------------------

rating_avg = df.groupby("rating")["score"].mean().sort_values(ascending=False).round(2)

print("=== Average Score by Age Rating ===")
print(rating_avg)
print()

# Top 10 Highest Rated Anime
# -------------------------------------------------------------------------------

top10 = df.nlargest(10, "score")[["name", "score", "type", "episodes"]]
print("=== Top 10 Highest Rated Anime ===")
print(top10.to_string(index=False))
print()


# Correlation Check
# relationship between number of voters and the score 
# does popularity (more voters) = higher quality
# ---------------------------------------------------------------------------------------

correlation = df["score"].corr(df["scored_by"])
print(f"=== Correlation: Score vs Number of Voters ===")
print(f"Correlation value: {correlation:.4f}")
print("(Closer to 1 = strong positive relationship, 0 = no relationship)\n")

# VISUALSSS
# ---------------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("What Makes an Anime Popular? — Trend Analysis", fontsize=16, fontweight="bold", y=1.01)

# Score Distribution (Histogram)
ax1 = axes[0, 0]
ax1.hist(df["score"], bins=30, color="#4C72B0", edgecolor="white")
ax1.set_title("Score Distribution Across All Anime")
ax1.set_xlabel("Score")
ax1.set_ylabel("Number of Anime")
ax1.axvline(df["score"].mean(), color="red", linestyle="--", label=f"Mean: {df['score'].mean():.2f}")
ax1.legend()
#----------------------------------------------------------------------------
# Average Score by Genre (top 10)
ax2 = axes[0, 1]
top10_genres = list(genre_avg_sorted.items())[:10]
genre_names = [g[0] for g in top10_genres]
genre_vals = [g[1] for g in top10_genres]
colors = ["#2ecc71" if v >= 7.0 else "#3498db" for v in genre_vals]
bars = ax2.barh(genre_names[::-1], genre_vals[::-1], color=colors[::-1], edgecolor="white")
ax2.set_title("Avg Score by Genre (min. 50 anime)")
ax2.set_xlabel("Average Score")
ax2.set_xlim(6.5, 7.5)
# Add value labels on the bars
for bar, val in zip(bars, genre_vals[::-1]):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2, str(val), va="center", fontsize=8)
#----------------------------------------------------------------------------

# Average Score by Episode Range
ax3 = axes[1, 0]
ep_order = ["1-12", "13-24", "25-50", "51-100", "100+"]
ep_vals = [ep_avg[ep] for ep in ep_order]
ax3.bar(ep_order, ep_vals, color="#e67e22", edgecolor="white")
ax3.set_title("Avg Score by Episode Count")
ax3.set_xlabel("Episode Range")
ax3.set_ylabel("Average Score")
ax3.set_ylim(6.0, 7.2)
for i, val in enumerate(ep_vals):
    ax3.text(i, val + 0.02, str(val), ha="center", fontsize=9)
#----------------------------------------------------------------------------

# Score vs Number of Voters (Scatter)
ax4 = axes[1, 1]
sample = df.dropna(subset=["score", "scored_by"]).sample(min(1000, len(df)), random_state=42)
ax4.scatter(sample["scored_by"], sample["score"], alpha=0.3, color="#9b59b6", s=10)
ax4.set_title(f"Score vs. Voters (corr = {correlation:.2f})")
ax4.set_xlabel("Number of People Who Scored")
ax4.set_ylabel("Score")
ax4.ticklabel_format(style="plain", axis="x")

plt.tight_layout()
plt.savefig("anime_analysis_charts.png", dpi=150, bbox_inches="tight")
print("Charts saved as 'anime_analysis_charts.png'")
plt.show()

# Findings
# --------------------------------------------------------------------------------

print("\n=== Summary of Findings ===")
print(f"- Total anime analyzed: {len(df)}")
print(f"- Average score across all anime: {df['score'].mean():.2f}")
print(f"- Highest scoring genre: {list(genre_avg_sorted.keys())[0]} ({list(genre_avg_sorted.values())[0]})")
print(f"- Best performing type: {type_avg.index[0]} (avg: {type_avg.iloc[0]:.2f})")
print(f"- Correlation between voters and score: {correlation:.4f}")
print("  --> More popular anime (more voters) tend to score slightly higher")
print("- Longer anime (100+ episodes) slightly outperform short ones")
print("- R-rated anime (17+) tend to score higher than G-rated ones")
