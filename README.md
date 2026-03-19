# Social Media Sentiment & Market Trend Analysis System

An end-to-end Python project for collecting social media text, running sentiment analysis, extracting product feedback themes, detecting market trends, and visualizing brand perception.

## What this project does

- Loads or collects 20,000+ social posts about a brand or product
- Cleans text and runs NLP preprocessing with spaCy
- Trains a sentiment classifier using TF-IDF + Logistic Regression
- Evaluates the model with accuracy, F1-score, classification report, and confusion matrix
- Extracts discussion themes with LDA topic modeling
- Detects daily or weekly sentiment shifts and trend spikes
- Generates charts and an optional Streamlit dashboard

## Project structure

```text
social_media_sentiment_market_trend_analysis/
│
├── data/
│   ├── raw/                     # place raw CSV files here
│   └── processed/               # generated cleaned data
├── notebooks/                   # EDA / experiments
├── src/
│   ├── collect.py               # load or collect data
│   ├── preprocess.py            # cleaning and spaCy preprocessing
│   ├── train.py                 # model training and evaluation
│   ├── topics.py                # topic modeling
│   ├── trends.py                # trend detection logic
│   ├── visualize.py             # charts and word clouds
│   ├── app.py                   # optional Streamlit dashboard
│   └── pipeline.py              # run the full pipeline
├── models/                      # saved vectorizer/model files
├── outputs/                     # metrics, charts, topic tables, trend files
├── requirements.txt
└── README.md
```

## Expected input data

This project supports two main workflows.

### Option 1: Pre-labeled sentiment training data
Use a CSV like Sentiment140 or your own labeled data with columns like:

- `text`
- `label` where values are negative / neutral / positive, or numeric labels that can be mapped
- `created_at` optional but recommended for trend plots
- `brand` optional

### Option 2: Brand-specific posts for analysis
Use a CSV with columns like:

- `text`
- `created_at`
- `brand`
- `source`

You can train on one dataset and score another.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Quick start

### 1. Train a sentiment model

```bash
python src/train.py \
  --input data/raw/sentiment_training.csv \
  --text-col text \
  --label-col label \
  --output-dir outputs \
  --model-dir models
```

### 2. Run end-to-end analysis on brand posts

```bash
python src/pipeline.py \
  --posts data/raw/brand_posts.csv \
  --model-path models/sentiment_model.joblib \
  --vectorizer-path models/tfidf_vectorizer.joblib \
  --output-dir outputs \
  --date-col created_at \
  --brand-col brand
```

### 3. Launch the dashboard

```bash
streamlit run src/app.py
```

## Recommended columns

| Column | Required | Purpose |
|---|---:|---|
| text | Yes | raw social post text |
| label | For training | sentiment class |
| created_at | For trends | time series analysis |
| brand | No | filter or compare brands |
| source | No | Twitter, Reddit, etc. |

## Output files

After running the project, you should see:

- `outputs/metrics.json`
- `outputs/classification_report.csv`
- `outputs/confusion_matrix.csv`
- `outputs/predicted_posts.csv`
- `outputs/topic_keywords.csv`
- `outputs/topic_assignments.csv`
- `outputs/daily_sentiment.csv`
- `outputs/trend_spikes.csv`
- `outputs/*.png` charts

## Notes on accuracy

The baseline pipeline uses TF-IDF + Logistic Regression because it is fast, interpretable, and often performs strongly on short text classification tasks. Reaching 85%+ accuracy depends heavily on data quality, label quality, class balance, and domain fit.

## Optional live collection

`src/collect.py` includes helpers for:

- loading CSV files
- filtering by brand keywords
- collecting Reddit submissions/comments with `praw` if credentials are available

Twitter/X API support is intentionally left as a simple placeholder because access rules can change. You can add your own authenticated fetcher in the same module.

## Stretch upgrades

- Replace the baseline model with a fine-tuned transformer
- Add NER with spaCy for brand and product extraction
- Expose sentiment scoring with FastAPI
- Dockerize the app
- Add multi-brand benchmarking

