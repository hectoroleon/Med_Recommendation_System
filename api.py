from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
from pydantic import BaseModel
import os
import gdown

# ==== 🔹 CONFIGURATION ====
MODEL_PATH = "models/cosine_sim_small.pkl"
CSV_PATH = "data/medicines_cleaned_small.csv"

# Google Drive file ID for the model
FILE_ID = "1tDShxVuPrZfEuu3VsbgB2J2ugAisrkZo"  

# ==== 🔹 FUNCTION TO DOWNLOAD THE MODEL ====
def download_model():
    """Download cosine_sim_small.pkl from Google Drive if it doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading cosine_sim_small.pkl from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        os.makedirs("models", exist_ok=True)  # Ensure directory exists
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully!")

# Ensure the model is available before proceeding
download_model()


# Define data types to reduce memory usage
dtypes = {
    "Medicine Name": "string",
    "Composition": "string",
    "Uses": "string",
    "Side_effects": "string",
    "Manufacturer": "string",
    "Satisfaction Score": "float32",
}

# Load dataset and precomputed similarity matrix
try:
    df = pd.read_csv("data/medicines_cleaned_small.csv", dtype=dtypes)  
    with open("models/cosine_sim_small.pkl", "rb") as f:
        cosine_sim = pickle.load(f)
except FileNotFoundError:
    raise Exception("❌ Error: 'medicines_cleaned_small.csv' or 'cosine_sim_small.pkl' not found.")

# Initialize FastAPI app
app = FastAPI()

# API request model
class MedicineRequest(BaseModel):
    medicine_name: str
    top_n: int = 5
    alpha: float = 0.8
    satisfaction_weight: float = 0.3
    side_effect_weight: float = 0.2
    manufacturer_weight: float = 0.3

# Home route
@app.get("/")
async def root():
    return {"message": "Welcome to the Medicine Recommendation API!"}

# Recommendation function
def recommend_medicines(medicine_name, df, cosine_sim, satisfaction_weight=0.3, side_effect_weight=0.2, manufacturer_weight=0.3, top_n=5, alpha=0.8):
    """
    Recommends medicines based on content similarity, user satisfaction, manufacturer reliability, and side effects.
    """
    idx = df[df["Medicine Name"].str.lower() == medicine_name.lower()].index
    if len(idx) == 0:
        raise HTTPException(status_code=404, detail="❌ Medicine not found.")

    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+10]  # Get top recommendations

    # Extract medicine indices
    med_indices = [i[0] for i in sim_scores]

    # Create recommendation DataFrame
    rec_df = df.iloc[med_indices][["Medicine Name", "Composition", "Uses", "Satisfaction Score", "Side_effects", "Manufacturer", "Manufacturer_Weight"]].copy()
    rec_df["Similarity"] = [i[1] for i in sim_scores]

    # Normalize Satisfaction Score
    rec_df["Satisfaction Score"] = (rec_df["Satisfaction Score"] - rec_df["Satisfaction Score"].min()) / \
                                    (rec_df["Satisfaction Score"].max() - rec_df["Satisfaction Score"].min())

    # Compute final score
    rec_df["Final Score"] = (
        (alpha * rec_df["Similarity"]) + 
        ((1 - alpha) * satisfaction_weight * rec_df["Satisfaction Score"]) - 
        (side_effect_weight * rec_df["Manufacturer_Weight"])
    )

    # Sort by final score and return top recommendations
    rec_df = rec_df.sort_values(by="Final Score", ascending=False).head(top_n)

    return rec_df.drop(columns=["Similarity", "Final Score", "Manufacturer_Weight"]).to_dict(orient="records")

# API endpoint to get medicine recommendations
@app.post("/recommend")
async def get_recommendations(request: MedicineRequest):
    return recommend_medicines(
        request.medicine_name,
        df,
        cosine_sim,
        request.satisfaction_weight,
        request.side_effect_weight,
        request.manufacturer_weight,
        request.top_n,
        request.alpha
    )

