import requests
import os

# ====================== CHECK IF FILES EXIST ======================
def check_files_exist():
    """Checks if all required CSV files already exist in the data directory."""
    all_exist = True
    for name in locations.keys():
        filename = f"nasa_power_data/{name.lower()}_meteorological_data.csv"
        if not os.path.exists(filename):
            all_exist = False
            break
    return all_exist

# ====================== CONFIG ======================
locations = {
    "Kano": {"lat": 12.0022, "lon": 8.5920},
    "Lagos": {"lat": 6.5244, "lon": 3.3792}
}

start_date = "20150101"      # 1 Jan 2015
end_date   = "20251231"      # 31 Dec 2025 (API returns latest available if future)
parameters = "ALLSKY_SFC_SW_DWN,WS50M,T2M"
community  = "RE"
base_url   = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# Create folder for data
os.makedirs("nasa_power_data", exist_ok=True)

if check_files_exist():
        print("\n📊 Data files already exist in 'nasa_power_data'. Skipping download.")
        # calling early exit to stop redownloading already existing data
        exit()


# ====================== FETCH DATA ======================
for name, coords in locations.items():
    url = (f"{base_url}?"
           f"parameters={parameters}&"
           f"community={community}&"
           f"longitude={coords['lon']}&"
           f"latitude={coords['lat']}&"
           f"start={start_date}&"
           f"end={end_date}&"
           f"format=CSV")
    
    print(f"📥 Fetching data for {name}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        filename = f"nasa_power_data/{name.lower()}_meteorological_data.csv"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"✅ Saved: {filename} ({len(response.text.splitlines())} lines)")
    else:
        print(f"❌ Error for {name}: {response.status_code}")

print("\n🎉 All done! Files are in the 'nasa_power_data' folder.")
print("Next step: open them in pandas and run your preprocessing pipeline.")