from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
from pydantic import BaseModel

# Load dataset and precomputed similarity matrix
try:
    df = pd.read_csv("data/medicines_cleaned.csv")  # Ensure this dataset is cleaned and updated
    with open("model/cosine_sim.pkl", "rb") as f:
        cosine_sim = pickle.load(f)
except FileNotFoundError:
    raise Exception("❌ Error: 'medicines_cleaned.csv' or 'cosine_sim.pkl' not found.")

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

