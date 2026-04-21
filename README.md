# NBA Game Prediction

A machine learning web app that predicts NBA game outcomes using real-time data from the NBA API.

## Features
- Live game predictions with win probabilities
- Ensemble model combining Logistic Regression, Random Forest, and Gradient Boosting
- Predicted vs actual stat comparison for final games
- Live box scores and player stats
- Back-to-back game and rest day tracking

## How It Works
1. Pulls today's games from the NBA live scoreboard API
2. Builds features from season stats, rolling form, rest days, and head-to-head history
3. Runs predictions through 3 models + an ensemble
4. Displays win probabilities and predicted margin for each game

## Stack
- Python, Flask
- scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
- nba_api
- pandas, numpy, joblib

## Run Locally
```bash
pip install -r requirements.txt
python app.py
```
App runs at `http://localhost:5000`

## Project Structure
```
app.py                  # Flask server + prediction routes
collect_data.py         # Pulls historical game data from NBA API
engineer_features.py    # Feature engineering pipeline
train_models.py         # Model training and evaluation
predict_today.py        # Standalone prediction script
templates/index.html    # Frontend UI
models/                 # Trained model files
data/                   # Raw and processed data
```
