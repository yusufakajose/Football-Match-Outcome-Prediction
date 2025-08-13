# Football Match Outcome Prediction

## Overview

Predicting football match outcomes is challenging due to small samples and many confounding factors. This project analyzes the Spanish La Liga 2011–2012 season and builds baseline machine learning models. It also adds an improved, reproducible modeling pipeline that avoids data leakage and can be extended with richer features.

## What's Inside

- EDA and feature experiments in the notebook `docs/football analysis v2.ipynb`
- Classic models (Decision Tree, Random Forest, SVM, Naive Bayes, KNN)
- New baseline: clean sklearn `Pipeline` with `OneHotEncoder`, 5-fold stratified CV, and holdout evaluation using only pre-match identifiers (`HomeTeam`, `AwayTeam`)

## Quickstart

1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Open the notebook

```bash
jupyter lab docs/"football analysis v2.ipynb"
```

3) Run the section "Improved, Reproducible Modeling Pipeline" at the end of the notebook.

## Data

- Source: `https://raw.githubusercontent.com/datasets/football-datasets/master/datasets/la-liga/season-1112.csv`
- Target: `FTR` with three classes: `H` (home win), `D` (draw), `A` (away win)

## Baseline Without Leakage

- Uses only pre-match identifiers (team names) with one-hot encoding
- 5-fold stratified cross-validation reports Accuracy and Macro-F1
- Final model refit on train and evaluated on a stratified 25% holdout

Why: in-match stats (shots, goals, fouls of the same fixture) are not available before kickoff. Using them to predict that fixture leaks outcome information.

## Roadmap (Improvements)

- Rolling features from past matches only (last N: goals for/against, shots, SOT, fouls, cards, corners)
- Recent form features (points in last 5, streaks)
- Simple Elo/SPI ratings; use rating differences as features
- Rest days and travel proxies
- Bookmaker odds, lineups/injuries where available
- Time-aware CV (e.g., `TimeSeriesSplit`) for chronological validation

## Results Snapshot

- Earlier experiments: accuracies roughly 45–60% depending on model and leakage risk
- New pipeline prints CV summary (Accuracy, Macro-F1) and holdout metrics for the best model

## License

This repository is for educational purposes. Check data source terms before redistribution.