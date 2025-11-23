import pandas as pd
import numpy as np

def multi_label_weather(condition):
    """
    Convert a single raw Weather_Condition string into multiple boolean flags.
    """

    flags = {
        "is_clear": 0,
        "is_cloudy": 0,
        "is_fog": 0,
        "is_rain": 0,
        "is_snow": 0,
        "is_storm": 0,
        "is_dust": 0,
        "is_windy": 0,
        "is_other": 0,
    }

    if pd.isna(condition):
        flags["is_other"] = 1
        return flags

    c = condition.lower()

    # Clear / Fair
    if any(word in c for word in ["clear", "fair", "sunny"]):
        flags["is_clear"] = 1

    # Cloudy / Overcast
    if any(word in c for word in ["cloud", "overcast"]):
        flags["is_cloudy"] = 1

    # Fog / Mist / Haze / Smoke
    if any(word in c for word in ["fog", "mist", "haze", "smoke"]):
        flags["is_fog"] = 1

    # Rain / Drizzle / Showers
    if any(word in c for word in ["rain", "drizzle", "shower", "wet"]):
        flags["is_rain"] = 1

    # Snow / Sleet / Ice / Freezing / Wintry Mix   <-- FIXED HERE
    if any(word in c for word in [
        "snow", "sleet", "ice", "freezing", "wintry",
        "snow grains", "blowing snow"
    ]):
        flags["is_snow"] = 1

    # Storm / Thunder / Hail / Lightning / Squalls
    if any(word in c for word in [
        "storm", "thunder", "t-storm", "lightning", "hail", "squall"
    ]):
        flags["is_storm"] = 1

    # Dust / Sand / Ash
    if any(word in c for word in ["dust", "sand", "ash"]):
        flags["is_dust"] = 1

    # Windy / Gusty / Breezy
    if any(word in c for word in ["wind", "gust", "breeze"]):
        flags["is_windy"] = 1

    # If none matched
    if sum(flags.values()) == 0:
        flags["is_other"] = 1

    return flags


def preprocess(df):

    # 1. FEATURES TO DROP ---------------------------------------
    ineffective_features = [
        'Distance(mi)', 'End_Time', 'Duration',
        'End_Lat', 'End_Lng', 'Description'
    ]

    # Precipitation(in) must NOT be dropped (paper uses NA feature + median)
    too_missing = ['Number', 'Wind_Chill(F)']

    drop_columns = [col for col in ineffective_features + too_missing if col in df.columns]
    df = df.drop(columns=drop_columns, errors='ignore')

    # 2. WIND DIRECTION CLEANING --------------------------------
    wind_map = {
        'CALM': 'CALM', 'Calm': 'CALM',
        'SW': 'SW', 'SSW': 'SW', 'WSW': 'SW',
        'S': 'S', 'SS': 'S', 'SSE': 'S',
        'W': 'W', 'WNW': 'W', 'WSW': 'W',
        'NW': 'NW', 'NNW': 'NW',
        'N': 'N', 'NNE': 'N',
        'VAR': 'VAR', 'Variable': 'VAR',
        'SE': 'SE', 'ESE': 'SE',
        'E': 'E', 'ENE': 'E',
        'NE': 'NE'
    }

    if 'Wind_Direction' in df.columns:
        df['Wind_Direction'] = df['Wind_Direction'].map(wind_map)

    # 3. DROP NA IN KEY COLUMNS --------------------------------
    for col in ['City', 'Zipcode', 'Airport_Code']:
        if col in df.columns:
            df = df[~df[col].isna()]

    # 4. PRECIPITATION NA FLAG + MEDIAN -------------------------
    if 'Precipitation(in)' in df.columns:
        df['Precipitation_NA'] = df['Precipitation(in)'].isna().astype(int)
        df['Precipitation(in)'] = df['Precipitation(in)'].fillna(df['Precipitation(in)'].median())

    # 5. MULTI-LABEL WEATHER ENCODING ---------------------------
    if 'Weather_Condition' in df.columns:
        weather_flags = df['Weather_Condition'].apply(multi_label_weather)
        weather_df = pd.DataFrame(list(weather_flags))
        df = df.drop(columns=['Weather_Condition'])
        df = pd.concat([df, weather_df], axis=1)

    # 6. IMPUTE NUMERIC COLUMNS --------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # 7. IMPUTE CATEGORICAL COLUMNS -----------------------------
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def main():
    df = pd.read_csv("Data/US_Accidents_March23.csv")
    print("Original Dataset Shape:", df.shape)

    df_clean = preprocess(df)
    print("After Preprocessing Shape:", df_clean.shape)

    df_clean.to_csv("Data/US_Accidents_Cleaned.csv", index=False)
    print("Saved cleaned dataset to Data/US_Accidents_Cleaned.csv")

if __name__ == "__main__":
    main()
