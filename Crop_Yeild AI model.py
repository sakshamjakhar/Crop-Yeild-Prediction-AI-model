import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import os

# === Load Datasets ===
temp_df = pd.read_csv(r"C:\Users\saksh\Downloads\Filtered_Temp_state.csv")
rainfall_df = pd.read_csv(r"D:\Sem X\ET-401\Crop-Yeild India\Sub_Division_IMD_2017_Rainfall.csv")
pest_df = pd.read_csv(r"C:\Users\saksh\Downloads\combined_pesticide_data.csv")
crop_df = pd.read_csv(r"C:\Users\saksh\Downloads\ICRISAT-District Level Data (1).csv")

# === Preprocess Temperature Data ===
temp_df = temp_df.rename(columns={"State": "State", "Year": "Year", "Temp": "Temp"})
temp_df["Year"] = temp_df["Year"].astype(int)

# === Preprocess Rainfall Data ===
rainfall_df = rainfall_df.rename(columns={"SUBDIVISION": "State", "YEAR": "Year", "ANNUAL": "Avg_Rainfall"})
rainfall_df = rainfall_df[["State", "Year", "Avg_Rainfall"]]
rainfall_df["Year"] = rainfall_df["Year"].astype(int)

# === Preprocess Pesticide Data ===
pest_df = pest_df.melt(id_vars=["State/UT"], var_name="Year", value_name="Avg_Pesticides")
pest_df = pest_df.rename(columns={"State/UT": "State"})
pest_df["Year"] = pest_df["Year"].str[:4].astype(int)

# === Preprocess Crop Production Data ===
crop_df = crop_df.rename(columns={
    "Dist Code": "Dist_Code",
    "Year": "Year",
    "State Code": "State_Code",
    "State Name": "State",
    "Dist Name": "District",
    "RICE PRODUCTION (1000 tons)": "Rice_Production",
    "WHEAT PRODUCTION (1000 tons)": "Wheat_Production",
    "MAIZE PRODUCTION (1000 tons)": "Maize_Production",
    "SUGARCANE PRODUCTION (1000 tons)": "Sugarcane_Production"
})
crop_df["Year"] = crop_df["Year"].astype(int)

# === Aggregate crop data per state-year ===
crops = {
    "rice": "Rice_Production",
    "wheat": "Wheat_Production",
    "maize": "Maize_Production",
    "sugarcane": "Sugarcane_Production"
}

# === Merge environmental data ===
env_df = pd.merge(temp_df, rainfall_df, on=["State", "Year"], how="inner")
env_df = pd.merge(env_df, pest_df, on=["State", "Year"], how="inner")

# === Train and save models for each crop ===
for crop_name, col in crops.items():
    crop_data = crop_df[["State", "Year", col]].copy()
    crop_data = crop_data.rename(columns={col: "Crop_Production"})
    
    merged = pd.merge(env_df, crop_data, on=["State", "Year"], how="inner")
    merged.dropna(inplace=True)

    if merged.empty:
        print(f"No data available for crop: {crop_name}")
        continue

    features = merged[["Temp", "Avg_Pesticides", "Avg_Rainfall"]]
    target = merged["Crop_Production"]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # === Calculate and print R² Score ===
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"{crop_name.capitalize()} Model R²: {r2:.2f}")

    joblib.dump(model, f"{crop_name}_production_model.pkl")

# === Prediction Function ===
def predict_crop_production(temp, pesticides, rainfall, crop):
    crop = crop.lower()
    model_path = f"{crop}_production_model.pkl"
    if not os.path.exists(model_path):
        return f"No model found for crop '{crop}'. Available crops: {list(crops.keys())}"

    model = joblib.load(model_path)
    input_data = pd.DataFrame([[temp, pesticides, rainfall]], columns=["Temp", "Avg_Pesticides", "Avg_Rainfall"])
    prediction = model.predict(input_data)[0]
    return prediction

# === User Input for Future Data ===
def get_user_input():
    state = input("Enter the state: ")
    year = int(input("Enter the year (e.g., 2030): "))
    temp = float(input("Enter the temperature (in Celsius): "))
    avg_pesticides = float(input("Enter the average pesticide use (in kg/hectare): "))
    avg_rainfall = float(input("Enter the average rainfall (in mm): "))
    crop = input("Enter the crop (e.g., rice, wheat, maize, sugarcane): ").lower()
    
    return {
        "State": state,
        "Year": year,
        "Temp": temp,
        "Avg_Pesticides": avg_pesticides,
        "Avg_Rainfall": avg_rainfall,
        "Crop": crop
    }

# === Unit Dictionary for Crop Production ===
crop_units = {
    "rice": "1000 tons", 
    "wheat": "1000 tons", 
    "maize": "1000 tons", 
    "sugarcane": "1000 tons"
}

# === Get User Input and Predict ===
user_input = get_user_input()

predicted_val = predict_crop_production(
    temp=user_input["Temp"],
    pesticides=user_input["Avg_Pesticides"],
    rainfall=user_input["Avg_Rainfall"],
    crop=user_input["Crop"]
)

# Add unit information to the output
crop_unit = crop_units.get(user_input["Crop"], "tons")
print(f"Predicted {user_input['Crop'].capitalize()} Production in {user_input['State']} ({user_input['Year']}): {predicted_val:.2f} {crop_unit}")
