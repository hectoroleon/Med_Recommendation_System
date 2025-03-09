import streamlit as st
import requests
import pandas as pd

# FastAPI endpoint URL
API_URL = "https://med-recommendation-system.onrender.com"

# Load the medicines dataset to provide autocomplete suggestions
df = pd.read_csv("data/medicines_cleaned_small.csv")  # Ensure this file is available

# Streamlit UI
st.title("💊 Medicine Recommendation System")
st.markdown("Find similar medicines based on composition, uses, side effects, and user satisfaction.")

# Sidebar - Explanation and Customization
st.sidebar.header("⚙️ Customize Your Recommendation")
st.sidebar.markdown("""
Fine-tune the recommendation system using these weights.  
**Recommended values are pre-selected**, but you can adjust them as needed.
""")

# Recommended values
RECOMMENDED_ALPHA = 0.8
RECOMMENDED_SATISFACTION_WEIGHT = 0.3
RECOMMENDED_SIDE_EFFECT_WEIGHT = 0.2
RECOMMENDED_MANUFACTURER_WEIGHT = 0.3

# Alpha (Similarity Weight)
alpha = st.sidebar.slider("⚖️ Similarity Weight (Alpha)", 0.0, 1.0, RECOMMENDED_ALPHA, step=0.05)
st.sidebar.markdown(f"""
🔹 **How Alpha Works:**  
- **Higher Alpha (closer to 1.0)** → Prioritizes medicines with **similar composition and usage**.  
- **Lower Alpha (closer to 0.0)** → Prioritizes **highly rated medicines**.  
**✅ Recommended Value: {RECOMMENDED_ALPHA}**
""")

# Satisfaction Weight
satisfaction_weight = st.sidebar.slider("⭐ Satisfaction Score Weight", 0.0, 1.0, RECOMMENDED_SATISFACTION_WEIGHT, step=0.05)
st.sidebar.markdown(f"""
🔹 **Satisfaction Score Weight:**  
- **Higher values** → Recommends medicines with **better user ratings**.  
- **Lower values** → Prioritizes similarity over satisfaction.  
**✅ Recommended Value: {RECOMMENDED_SATISFACTION_WEIGHT}**
""")

# Side Effect Penalty
side_effect_weight = st.sidebar.slider("⚠️ Side Effect Penalty", 0.0, 1.0, RECOMMENDED_SIDE_EFFECT_WEIGHT, step=0.05)
st.sidebar.markdown(f"""
🔹 **Side Effect Penalty:**  
- **Higher values** → Avoids recommending medicines with **similar adverse effects**.  
- **Lower values** → Allows medicines with similar side effects to be recommended.  
**✅ Recommended Value: {RECOMMENDED_SIDE_EFFECT_WEIGHT}**
""")

# Manufacturer Penalty
manufacturer_weight = st.sidebar.slider("🏭 Manufacturer Penalty", 0.0, 1.0, RECOMMENDED_MANUFACTURER_WEIGHT, step=0.05)
st.sidebar.markdown(f"""
🔹 **Manufacturer Penalty:**  
- **Higher values** → **Reduces recommendations from rare manufacturers**, favoring trusted brands.  
- **Lower values** → Allows more recommendations from smaller manufacturers.  
**✅ Recommended Value: {RECOMMENDED_MANUFACTURER_WEIGHT}**
""")

# Number of recommendations
top_n = st.sidebar.slider("🔢 Number of Recommendations", 1, 10, 5)

# Medicine selection with autocomplete dropdown
medicine_name = st.selectbox(
    "🔍 Select a medicine:",
    options=df["Medicine Name"].unique(),
    index=0
)

# Submit button
if st.button("🔎 Get Recommendations"):
    # API request payload
    payload = {
        "medicine_name": medicine_name,
        "top_n": top_n,
        "alpha": alpha,
        "satisfaction_weight": satisfaction_weight,
        "side_effect_weight": side_effect_weight,
        "manufacturer_weight": manufacturer_weight
    }

    # Send request to FastAPI
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        recommendations = response.json()

        # Display recommendations
        st.subheader(f"📌 Recommended Medicines for: {medicine_name}")

        if recommendations:
            for idx, rec in enumerate(recommendations, start=1):
                st.markdown(f"""
                **{idx}. {rec['Medicine Name']}**  
                - **Composition:** {rec['Composition']}
                - **Uses:** {rec['Uses']}
                - **Satisfaction Score:** {round(rec['Satisfaction Score'], 2)}
                - **Side Effects:** {rec['Side_effects']}
                - **Manufacturer:** {rec['Manufacturer']}
                """)
        else:
            st.warning("⚠️ No recommendations found. Try adjusting the parameters.")
    else:
        st.error("❌ Error fetching recommendations. Please check the API connection.")

# Footer - Developer Credit
st.markdown("---")
st.markdown("🚀 Developed by **Héctor Ornelas**")
st.markdown("Built with [Streamlit](https://streamlit.io/) & [FastAPI](https://fastapi.tiangolo.com/)")
