import pandas as pd

# load data
df = pd.read_csv("data/raw/amazon_reviews.csv")

# keep only what we need
df = df[
    [
        "reviews.text",
        "reviews.rating",
        "reviews.date",
        "brand",
        "name",
    ]
].dropna()

# clean text fields
for col in ["reviews.text", "brand", "name"]:
    df[col] = df[col].astype(str).str.strip()

# remove empty rows after stripping
df = df[
    (df["reviews.text"] != "")
    & (df["brand"] != "")
    & (df["name"] != "")
].copy()

# normalize rating
df["reviews.rating"] = pd.to_numeric(df["reviews.rating"], errors="coerce")
df = df[df["reviews.rating"].notna()].copy()

# parse review date
df["reviews.date"] = pd.to_datetime(df["reviews.date"], errors="coerce", utc=True)
df = df[df["reviews.date"].notna()].copy()

# normalize brand names
df["brand"] = df["brand"].str.lower().str.strip()
df["brand"] = df["brand"].replace(
    {
        "amazonbasics": "AmazonBasics",
        "amazon basics": "AmazonBasics",
        "amazon": "Amazon",
    }
)

# map rating -> sentiment
def map_rating(r):
    if r >= 4:
        return "positive"
    elif r <= 2:
        return "negative"
    else:
        return "neutral"

df["label"] = df["reviews.rating"].apply(map_rating)

# rename for pipeline
df = df.rename(
    columns={
        "reviews.text": "text",
        "reviews.date": "created_at",
    }
)

# optional deduplication
df = df.drop_duplicates(subset=["text", "created_at", "brand", "name"]).copy()

# final dataset
df = df[["text", "label", "created_at", "brand", "name"]]

# save
df.to_csv("data/raw/amazon_training_data.csv", index=False)

print("Amazon dataset prepared successfully")
print(f"Rows saved: {len(df):,}")
print("Brand counts:")
print(df["brand"].value_counts())