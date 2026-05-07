# NBA Game Prediction

**Live Site:** [therodma.github.io/NBA_Winner_Prediction](https://therodma.github.io/NBA_Winner_Prediction)

A machine learning web app that predicts NBA game outcomes using real-time data from the NBA API.

## Features
- Live game predictions with win probabilities
- Ensemble model combining Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
- Win probability line graph (Kalshi-style) that updates in real time during live games
- Injury report with Out / Doubtful / Questionable player statuses
- Two-scenario predictions for questionable players — if they play vs if they sit
- Predicted vs actual stat comparison for final games
- Live box scores and player stats
- Back-to-back game and rest day tracking
- Yesterday's results with correct/wrong badges and record
- Refreshes every 10 seconds, backend cache updates every 60 seconds

## How It Works
1. Pulls today's games from the NBA live scoreboard API
2. Fetches injury report and player availability
3. Builds features from season stats, rolling form, rest days, home court strength, travel/timezone, and H2H history
4. Runs predictions through 4 models + an ensemble
5. Applies injury adjustments to win probabilities
6. Generates play/sit scenarios for questionable players
7. Displays win probabilities, predicted margin, and live probability trend chart

## Models
- Logistic Regression
- Random Forest (500 estimators)
- Gradient Boosting (500 estimators)
- XGBoost (500 estimators)
- Soft-voting Ensemble of all four

Trained on 10 seasons (2015-16 to 2024-25) with time-weighted samples — recent seasons count more.

## Data Sources
- **Season stats**: Off/Def/Net Rating, Pace, Win%, FG%, 3P%, FT%, Reb, Ast, Tov, Stl, Blk, TS%, AST%, REB%, TOV%
- **Rolling form**: Last 10 and 20 game win% and point differential
- **Streak**: Current win/loss streak
- **Rest & fatigue**: Days of rest, back-to-back flag
- **Home court strength**: Each team's home win% minus away win% per season
- **Travel**: Timezone difference for the away team
- **Head-to-head**: Home team win% in last 20 matchups
- **Injury report**: Player availability (Out, Doubtful, Questionable) with impact-based probability adjustments

## Stack
- Python, Flask
- scikit-learn, XGBoost
- nba_api
- pandas, numpy, joblib
- APScheduler
- Chart.js (frontend probability graph)
- GitHub Pages (frontend) + Render (backend)
- GitHub Gist (persistent history storage)

## Run Locally
```bash
pip install -r requirements.txt
python app.py
```
App runs at `http://localhost:5000`

## Retrain Models
```bash
python collect_data.py       # ~15-20 min, pulls 10 seasons from NBA API
python engineer_features.py  # builds feature matrix
python train_models.py       # trains and saves models to models/
```

## Project Structure
```
app.py                  # Flask server + prediction routes + scheduler
collect_data.py         # Pulls historical game data from NBA API
engineer_features.py    # Feature engineering pipeline
train_models.py         # Model training and evaluation
predict_today.py        # Standalone prediction script
index.html              # GitHub Pages static frontend
templates/index.html    # Flask template (local use)
models/                 # Trained model files
data/                   # Raw and processed data
```
