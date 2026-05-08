# Anime Popularity Trend Analysis
**CS 210 — Data Management for Data Science | Rutgers University | Spring 2026**

## Overview
This project analyzes trends determining anime popularity using the MyAnimeList dataset (28,825 entries) found through Kaggle.
Dataset itself should be attached in file along with document with link to video and github.

## Files
- `anime_popularity_analysis.py` —> main analysis script with entire python code
- `popular_anime.csv` —> dataset from MyAnimeList found through Kaggle

## How to Run
1. Make sure pandas, matplotlib, and all the other packages are downloaded and installed: `pip install pandas matplotlib`
2. Place both files in the same folder
3. Run: `python anime_popularity_analysis.py`

## Key Findings (Average Scores in (n))
- Top rated animes: Frieren:Beyond Journey's End (9.30), FMA Brotherhood (9.10), Steins;Gate (9.07)
- Best genre: Suspense (7.08) / Mystery (7.01)
- Best format: TV series (6.84) 
- Best studio: Kyoto Animation (7.45)
- Best Age Rating: R 17+ (6.96)

- Weakest predictors: Episode Count (0.06) and duration (0.15) have weak correlations 
- Strongest predictor: voter count (correlation = 0.35)
