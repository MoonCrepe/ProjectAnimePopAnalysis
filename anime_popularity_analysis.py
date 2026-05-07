# Anime Popularity Trend Analysis
# CS 210 - Data Management for Data Science
# Goal: Find what trends/factors influence anime popularity (score)
# Dataset: popular_anime.csv (from MyAnimeList)

# -----------------------------------------------
# Imports
# pandas  -> data manipulation (Week 04)
# matplotlib -> visualization (Week 06)
# defaultdict -> dictionary that never throws KeyError (Week 02/03)
# re -> regex for parsing duration strings (Week 09)
# -----------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
import re

# -----------------------------------------------
# Load data
# -----------------------------------------------

df = pd.read_csv("popular_anime.csv")

print(f"Total rows:   {len(df)}")
print(f"Columns:      {list(df.columns)}\n")

# -----------------------------------------------
# Clean the data
# Drop rows missing the columns we care most about,
# then convert everything to the right data types
# -----------------------------------------------

df = df.dropna(subset=["score", "rank", "scored_by"])
df = df.reset_index(drop=True)

df["score"]     = pd.to_numeric(df["score"],     errors="coerce")
df["episodes"]  = pd.to_numeric(df["episodes"],  errors="coerce")
df["scored_by"] = pd.to_numeric(df["scored_by"], errors="coerce")

# Parse the year out of the aired_from column
# pd.to_datetime turns "2011-04-06T00:00:00+00:00" into a real date object
df["year"] = pd.to_datetime(df["aired_from"], errors="coerce").dt.year

# Parse duration string like "24 min per ep" into just the number 24
# re.search looks for the first number in the string (Week 09 - Regex)
def parse_duration(val):
    if pd.isna(val):
        return None
    match = re.search(r"(\d+)", str(val))
    return int(match.group(1)) if match else None

df["duration_min"] = df["duration_per_ep"].apply(parse_duration)

print(f"Rows after cleaning: {len(df)}\n")

# -----------------------------------------------
# Exclude R18 content
#  filter out "Rx - Hentai" so  analysis focuses on
# mainstream anime only ->  makes the findings more relevant
# -----------------------------------------------


df_all    = df.copy()                                         
df        = df[df["rating"] != "Rx - Hentai"].reset_index(drop=True)

print(f"Rows after removing R18 content: {len(df)}")
print(f"(Removed {len(df_all) - len(df)} R18 rows)\n")

# -----------------------------------------------
# Basic stats  
# -----------------------------------------------

print("=== Basic Score Statistics (R18 Excluded) ===")
print(df["score"].describe().round(2))
print()

print("=== Anime Types in Dataset ===")
print(df["type"].value_counts())
print()

print("=== Status Breakdown ===")
print(df["status"].value_counts())
print()

# -----------------------------------------------
# Genre Analysis 
# Each anime can have multiple genres separated by commas
#  use a defaultdict(list) to collect scores per genre,
# then compute averages and sort dictionary
# -----------------------------------------------

genre_scores = defaultdict(list)

for _, row in df.iterrows():
    if pd.notna(row["genres"]):
        for genre in row["genres"].split(","):
            genre = genre.strip()
            if genre:
                genre_scores[genre].append(row["score"])

# Only keep genres with at least 50 anime so the average is reliable
genre_avg = {}
for genre, scores in genre_scores.items():
    if len(scores) >= 50:
        genre_avg[genre] = round(sum(scores) / len(scores), 2)

genre_avg_sorted = dict(sorted(genre_avg.items(), key=lambda x: x[1], reverse=True))

print("=== Average Score by Genre (top 10) ===")
for genre, avg in list(genre_avg_sorted.items())[:10]:
    print(f"  {genre}: {avg}")
print()

# -----------------------------------------------
# Groupby analyses 
# Split-Apply-Combine: split by category, apply mean, combine results
# -----------------------------------------------

# -- By type (TV, Movie, OVA...)
type_avg = df.groupby("type")["score"].mean().sort_values(ascending=False)
print("=== Average Score by Type ===")
print(type_avg.round(2))
print()

# -- By age rating
rating_avg = df.groupby("rating")["score"].mean().sort_values(ascending=False)
print("=== Average Score by Age Rating ===")
print(rating_avg.round(2))
print()

# -- By airing status
status_avg = df.groupby("status")["score"].mean().sort_values(ascending=False)
print("=== Average Score by Status ===")
print(status_avg.round(2))
print()

# -----------------------------------------------
# Episode bins  
# pd.cut puts continuous values into labeled buckets
# -----------------------------------------------

bins   = [0, 12, 24, 50, 100, 10000]
labels = ["1-12", "13-24", "25-50", "51-100", "100+"]
df["episode_range"] = pd.cut(df["episodes"], bins=bins, labels=labels)
ep_avg = df.groupby("episode_range", observed=True)["score"].mean().round(2)

print("=== Average Score by Episode Count ===")
print(ep_avg)
print()

# -----------------------------------------------
# Duration bins 
# -----------------------------------------------

dur_bins   = [0, 5, 15, 25, 35, 999]
dur_labels = ["<5 min", "5-15 min", "15-25 min", "25-35 min", "35+ min"]
df["duration_range"] = pd.cut(df["duration_min"], bins=dur_bins, labels=dur_labels)
dur_avg = df.groupby("duration_range", observed=True)["score"].mean().round(2)

print("=== Average Score by Episode Duration ===")
print(dur_avg)
print()

# -----------------------------------------------
# Year trend 
# Filter to 2000-2024 and look at how average scores changed over time
# -----------------------------------------------

year_df  = df[(df["year"] >= 2000) & (df["year"] <= 2024)]
year_avg = year_df.groupby("year")["score"].mean().round(2)
year_cnt = year_df.groupby("year")["score"].count()

print("=== Average Score by Year (2000-2024, sample) ===")
print(year_avg.tail(10))
print()

# -----------------------------------------------
# Studio analysis
# Pull the first listed studio for each anime and group by it
# -----------------------------------------------

studio_scores = defaultdict(list)

for _, row in df.iterrows():
    if pd.notna(row["studios"]):
        # Just take the first studio listed
        studio = row["studios"].split(",")[0].strip()
        if studio:
            studio_scores[studio].append(row["score"])

# Only keep studios with at least 30 anime
studio_avg = {}
for studio, scores in studio_scores.items():
    if len(scores) >= 30:
        studio_avg[studio] = round(sum(scores) / len(scores), 2)

studio_avg_sorted = dict(sorted(studio_avg.items(), key=lambda x: x[1], reverse=True))

print("=== Top 10 Studios by Avg Score ===")
for studio, avg in list(studio_avg_sorted.items())[:10]:
    print(f"  {studio}: {avg}")
print()

# -----------------------------------------------
# Correlation check  (Week 08 - Data Modeling)
# .corr() measures the linear relationship between two columns
# 0 = no relationship, 1 = perfect positive, -1 = perfect negative
# -----------------------------------------------

corr_voters   = df["score"].corr(df["scored_by"])
corr_episodes = df["score"].corr(df["episodes"])
corr_duration = df["score"].corr(df["duration_min"])
corr_year     = df["score"].corr(df["year"])

print("=== Correlation with Score ===")
print(f"  Voters (scored_by): {corr_voters:.4f}   --> weak-moderate positive influence")
print(f"  Episode count:      {corr_episodes:.4f}  --> very weak, not a strong influence")
print(f"  Duration (min/ep):  {corr_duration:.4f}  --> very weak, not a strong influence")
print(f"  Release year:       {corr_year:.4f}   --> slight positive, newer = slightly higher")
print()

# -----------------------------------------------
# Top 10 Highest Rated Anime
#  -----------------------------------------------
top10 = df.nlargest(10, "score")[["name", "score", "type", "episodes", "rating"]]
print("=== Top 10 Highest Rated Anime (R18 Excluded) ===")
print(top10.to_string(index=False))
print()

# -----------------------------------------------
# Write findings to a file  (Week 03 - File Processing)
# open() with "w" creates or overwrites the file
# -----------------------------------------------

with open("anime_findings.txt", "w") as f:
    f.write("=== Anime Popularity Analysis — Findings ===\n\n")
    f.write(f"Total anime analyzed (R18 excluded): {len(df)}\n")
    f.write(f"Average score: {df['score'].mean():.2f}\n\n")
    f.write("-- Correlations with Score --\n")
    f.write(f"  Voters:    {corr_voters:.4f}  (weak-moderate influence -- YES)\n")
    f.write(f"  Episodes:  {corr_episodes:.4f}  (very weak -- NOT a strong influence)\n")
    f.write(f"  Duration:  {corr_duration:.4f}  (very weak -- NOT a strong influence)\n")
    f.write(f"  Year:      {corr_year:.4f}  (slight positive -- minor influence)\n\n")
    f.write("-- Top Genre --\n")
    top_genre = list(genre_avg_sorted.items())[0]
    f.write(f"  {top_genre[0]}: {top_genre[1]}\n\n")
    f.write("-- Top Studio --\n")
    top_studio = list(studio_avg_sorted.items())[0]
    f.write(f"  {top_studio[0]}: {top_studio[1]}\n")

print("Findings written to anime_findings.txt\n")

# -----------------------------------------------
# Visualizations 
# -----------------------------------------------

# helper so we don't repeat plt.tight_layout + savefig every time
def save_fig(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")

# ── Figure 1: Score Distribution Histogram ─────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["score"], bins=35, color="#4C72B0", edgecolor="white")
ax.axvline(df["score"].mean(),   color="red",    linestyle="--", linewidth=1.5,
           label=f"Mean:   {df['score'].mean():.2f}")
ax.axvline(df["score"].median(), color="orange", linestyle="--", linewidth=1.5,
           label=f"Median: {df['score'].median():.2f}")
ax.set_title("Fig 1 — Score Distribution (R18 Excluded)", fontsize=13, fontweight="bold")
ax.set_xlabel("Score"); ax.set_ylabel("Number of Anime")
ax.legend()
save_fig("fig1_score_distribution.png")

# ── Figure 2: R18 vs Non-R18 score comparison ──────────────────────────────
# This shows WHY we excluded R18 -- they score much lower on average
r18_scores    = df_all[df_all["rating"] == "Rx - Hentai"]["score"].dropna()
non_r18_scores = df["score"].dropna()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Fig 2 — R18 vs Non-R18 Score Comparison", fontsize=13, fontweight="bold")
axes[0].hist(non_r18_scores, bins=30, color="#4C72B0", edgecolor="white", alpha=0.8, label="Non-R18")
axes[0].hist(r18_scores,     bins=30, color="#EF5350", edgecolor="white", alpha=0.6, label="R18 (Hentai)")
axes[0].axvline(non_r18_scores.mean(), color="blue",   linestyle="--", label=f"Non-R18 mean: {non_r18_scores.mean():.2f}")
axes[0].axvline(r18_scores.mean(),     color="red",    linestyle="--", label=f"R18 mean:     {r18_scores.mean():.2f}")
axes[0].set_xlabel("Score"); axes[0].set_ylabel("Count")
axes[0].set_title("Score Distributions Overlaid"); axes[0].legend(fontsize=8)

# Box plot comparison
axes[1].boxplot([non_r18_scores, r18_scores],
                labels=["Non-R18", "R18 (Hentai)"],
                patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", alpha=0.6))
axes[1].set_ylabel("Score")
axes[1].set_title("Box Plot Comparison")
save_fig("fig2_r18_vs_nonr18.png")

# ── Figure 3: Average Score by Genre (top 15, horizontal bar) ──────────────
top15_genres = list(genre_avg_sorted.items())[:15]
g_names = [x[0] for x in top15_genres]
g_vals  = [x[1] for x in top15_genres]
colors  = ["#2ecc71" if v >= 7.0 else "#3498db" for v in g_vals]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(g_names[::-1], g_vals[::-1], color=colors[::-1], edgecolor="white")
for bar, val in zip(bars, g_vals[::-1]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=8)
ax.set_xlim(6.4, 7.5)
ax.set_xlabel("Average Score")
ax.set_title("Fig 3 — Avg Score by Genre (min. 50 anime, green = 7.0+)",
             fontsize=12, fontweight="bold")
save_fig("fig3_genre_scores.png")

# ── Figure 4: Score by Anime Type (bar chart) ──────────────────────────────
# Only show types with at least 20 anime
type_counts = df.groupby("type")["score"].count()
valid_types = type_counts[type_counts >= 20].index
type_avg_filtered = df[df["type"].isin(valid_types)].groupby("type")["score"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(type_avg_filtered.index, type_avg_filtered.values,
              color="#5C85D6", edgecolor="white")
for bar, val in zip(bars, type_avg_filtered.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", fontsize=9)
ax.set_ylim(5.5, 7.5)
ax.set_ylabel("Average Score")
ax.set_title("Fig 4 — Avg Score by Anime Type", fontsize=12, fontweight="bold")
save_fig("fig4_type_scores.png")

# ── Figure 5: Score by Age Rating (bar chart) ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
rating_avg_filtered = rating_avg.drop("Rx - Hentai", errors="ignore")   # already excluded but just in case
short_labels = {
    "G - All Ages":                    "G\nAll Ages",
    "PG - Children":                   "PG\nChildren",
    "PG-13 - Teens 13 or older":       "PG-13\nTeens",
    "R - 17+ (violence & profanity)":  "R\n17+",
    "R+ - Mild Nudity":                "R+\nMild Nudity",
}
labels_short = [short_labels.get(r, r) for r in rating_avg_filtered.index]
bars = ax.bar(labels_short, rating_avg_filtered.values, color="#E67E22", edgecolor="white")
for bar, val in zip(bars, rating_avg_filtered.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", fontsize=9)
ax.set_ylim(5.5, 7.5)
ax.set_ylabel("Average Score")
ax.set_title("Fig 5 — Avg Score by Age Rating (R18 Excluded)", fontsize=12, fontweight="bold")
save_fig("fig5_rating_scores.png")

# ── Figure 6: Score by Episode Count (bar chart) ───────────────────────────
ep_order = ["1-12", "13-24", "25-50", "51-100", "100+"]
ep_vals  = [ep_avg[ep] for ep in ep_order]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(ep_order, ep_vals, color="#9B59B6", edgecolor="white")
for i, val in enumerate(ep_vals):
    ax.text(i, val + 0.02, str(val), ha="center", fontsize=9)
ax.set_ylim(6.0, 7.3)
ax.set_xlabel("Episode Range"); ax.set_ylabel("Average Score")
ax.set_title("Fig 6 — Avg Score by Episode Count", fontsize=12, fontweight="bold")
save_fig("fig6_episode_scores.png")

# ── Figure 7: Score by Episode Duration ────────────────────────────────────
dur_order  = ["<5 min", "5-15 min", "15-25 min", "25-35 min", "35+ min"]
dur_vals   = [dur_avg[d] if d in dur_avg.index else 0 for d in dur_order]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(dur_order, dur_vals, color="#1ABC9C", edgecolor="white")
for i, val in enumerate(dur_vals):
    ax.text(i, val + 0.02, str(val), ha="center", fontsize=9)
ax.set_ylim(5.0, 7.5)
ax.set_xlabel("Episode Duration"); ax.set_ylabel("Average Score")
ax.set_title("Fig 7 — Avg Score by Episode Duration", fontsize=12, fontweight="bold")
save_fig("fig7_duration_scores.png")

# ── Figure 8: Score trend over years (dual axis -- line + bar) ─────────────
fig, ax1 = plt.subplots(figsize=(13, 5))
ax2 = ax1.twinx()   # second y-axis on the right side

ax1.plot(year_avg.index, year_avg.values, color="#3F51B5",
         linewidth=2, marker="o", markersize=4, label="Avg Score")
ax1.fill_between(year_avg.index, year_avg.values, alpha=0.1, color="#3F51B5")
ax2.bar(year_cnt.index, year_cnt.values, alpha=0.25, color="#78909C", label="# Titles")

ax1.set_xlabel("Release Year")
ax1.set_ylabel("Average Score", color="#3F51B5")
ax2.set_ylabel("Number of Titles", color="#78909C")
ax1.set_title("Fig 8 — Avg Score & Title Count by Year (2000-2024)",
              fontsize=12, fontweight="bold")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")
save_fig("fig8_score_over_years.png")

# ── Figure 9: Top 12 Studios by avg score (horizontal bar) ─────────────────
top12_studios = list(studio_avg_sorted.items())[:12]
s_names = [x[0] for x in top12_studios]
s_vals  = [x[1] for x in top12_studios]

fig, ax = plt.subplots(figsize=(10, 6))
colors_s = ["#F39C12" if v >= 7.0 else "#85C1E9" for v in s_vals]
bars = ax.barh(s_names[::-1], s_vals[::-1], color=colors_s[::-1], edgecolor="white")
for bar, val in zip(bars, s_vals[::-1]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=8)
ax.set_xlim(6.4, 8.0)
ax.set_xlabel("Average Score")
ax.set_title("Fig 9 — Top 12 Studios by Avg Score (min. 30 titles, gold = 7.0+)",
             fontsize=12, fontweight="bold")
save_fig("fig9_studio_scores.png")

# ── Figure 10: Score vs Number of Voters (scatter) ─────────────────────────
sample = df.dropna(subset=["score", "scored_by"]).sample(min(1500, len(df)), random_state=42)

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(sample["scored_by"], sample["score"], alpha=0.25, color="#9B59B6", s=10)
ax.set_title(f"Fig 10 — Score vs Voters  (correlation = {corr_voters:.2f})",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Number of People Who Scored")
ax.set_ylabel("Score")
ax.ticklabel_format(style="plain", axis="x")
save_fig("fig10_score_vs_voters.png")

# ── Figure 11: Score vs Episode Count (scatter) ────────────────────────────
ep_sample = df.dropna(subset=["score", "episodes"])
ep_sample = ep_sample[ep_sample["episodes"] <= 200]   # cap outliers for readability

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(ep_sample["episodes"], ep_sample["score"], alpha=0.2, color="#2ECC71", s=10)
ax.set_title(f"Fig 11 — Score vs Episode Count  (correlation = {corr_episodes:.2f})",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Number of Episodes (capped at 200)")
ax.set_ylabel("Score")
save_fig("fig11_score_vs_episodes.png")

# ── Figure 12: Score by Duration (scatter) ─────────────────────────────────
dur_sample = df.dropna(subset=["score", "duration_min"])
dur_sample = dur_sample[dur_sample["duration_min"] <= 60]

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(dur_sample["duration_min"], dur_sample["score"], alpha=0.2, color="#E74C3C", s=10)
ax.set_title(f"Fig 12 — Score vs Episode Duration  (correlation = {corr_duration:.2f})",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Episode Duration (minutes, capped at 60)")
ax.set_ylabel("Score")
save_fig("fig12_score_vs_duration.png")

# ── Figure 13: Correlation bar chart summary ────────────────────────────────
# This is a clean summary of which factors actually influenced score
factors      = ["Voters", "Release Year", "Episodes", "Duration"]
correlations = [corr_voters, corr_year, corr_episodes, corr_duration]
bar_colors   = ["#2ECC71" if abs(c) >= 0.2 else "#E74C3C" for c in correlations]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(factors, correlations, color=bar_colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline( 0.2, color="gray", linestyle="--", linewidth=0.8, label="±0.2 threshold")
ax.axhline(-0.2, color="gray", linestyle="--", linewidth=0.8)
for bar, val in zip(bars, correlations):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + 0.005 if val >= 0 else val - 0.015,
            f"{val:.3f}", ha="center", fontsize=9)
ax.set_ylabel("Correlation with Score")
ax.set_title("Fig 13 — Which Factors Influence Anime Score?\n(Green = notable influence  |  Red = weak/no influence)",
             fontsize=11, fontweight="bold")
ax.legend()
save_fig("fig13_correlation_summary.png")

# ── Figure 14: Genre count (how many anime per genre) ──────────────────────
# Checks if certain genres just have more anime -- size isn't the same as quality
genre_counts = {g: len(s) for g, s in genre_scores.items() if len(s) >= 50}
genre_counts_sorted = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True))
top_genre_counts = list(genre_counts_sorted.items())[:15]
gc_names = [x[0] for x in top_genre_counts]
gc_vals  = [x[1] for x in top_genre_counts]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(gc_names[::-1], gc_vals[::-1], color="#85C1E9", edgecolor="white")
ax.set_xlabel("Number of Anime")
ax.set_title("Fig 14 — Most Common Genres (by title count)",
             fontsize=12, fontweight="bold")
save_fig("fig14_genre_counts.png")

# ==============================================================
# Summary 
# ==============================================================

print("\n" + "=" * 58)
print("     SUMMARY OF FINDINGS")
print("=" * 58)
print(f"  Total anime analyzed (R18 excluded): {len(df)}")
print(f"  Removed R18 entries:                 {len(df_all) - len(df)}")
print(f"  Average score:                       {df['score'].mean():.2f}")
print()
print("  -- Did it influence popularity? --")
print(f"  Voters (scored_by):  {corr_voters:.3f}  --> YES, weak-moderate positive link")
print(f"  Release year:        {corr_year:.3f}  --> SLIGHT, newer anime score marginally higher")
print(f"  Episode count:       {corr_episodes:.3f}  --> NO, barely any relationship")
print(f"  Episode duration:    {corr_duration:.3f}  --> NO, barely any relationship")
print()
print("  -- Categorical trends --")
print(f"  Best genre:   {list(genre_avg_sorted.keys())[0]} ({list(genre_avg_sorted.values())[0]})")
print(f"  Best type:    {type_avg.index[0]} ({type_avg.iloc[0]:.2f})")
print(f"  Best rating:  {rating_avg_filtered.index[0]} ({rating_avg_filtered.iloc[0]:.2f})")
print(f"  Best studio:  {list(studio_avg_sorted.keys())[0]} ({list(studio_avg_sorted.values())[0]})")
print()
print("  Charts saved: fig1 through fig14")
print("=" * 58)
