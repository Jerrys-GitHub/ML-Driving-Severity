import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pyarrow

# 1. ADD TIME FEATURES
def add_time_features(df):
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce", format="mixed")

    #standard times
    df["hour"] = df["Start_Time"].dt.hour
    df["weekday"] = df["Start_Time"].dt.weekday
    df["month"] = df["Start_Time"].dt.month
    #0 is monday, 5-6 are saturday/sunday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    #7:00am to 10:00am and 4:00pm to 7:00pm
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

    return df

# 2. ADD DAYLIGHT FEATURE
def add_daylight_feature(df):
    df["is_daylight"] = (df["Sunrise_Sunset"] == "Day").astype(int)
    return df

# 3. BINARY SEVERITY
def add_binary_severity(df):
    df["Severity_Binary"] = (df["Severity"] >= 3).astype(int)
    return df

# 4. DROP UNUSED COLUMNS
def drop_unused(df):
    cols_to_drop = [
        "ID", "Source", "Airport_Code", "Street", "City", "County",
        "Zipcode", "Country", "Timezone",
        "End_Time", "End_Lat", "End_Lng",
        "Weather_Timestamp",
        "Sunrise_Sunset", "Civil_Twilight",
        "Nautical_Twilight", "Astronomical_Twilight",
        "Start_Time"
    ]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

# 4. Convert columns to boolean 0/1 values.
def fix_boolean_columns(df):
    # Convert actual boolean dtype → int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Convert float columns that are only {0.0, 1.0} → int
    for col in df.columns:
        if df[col].dtype == float:
            unique_vals = set(df[col].dropna().unique())
            if unique_vals.issubset({0.0, 1.0}):
                df[col] = df[col].astype(int)

    return df

# 5. FULL SECOND-STAGE PREPROCESS OF STEPS 1-4
def preprocess(df):
    df = add_time_features(df)
    df = add_daylight_feature(df)
    df = add_binary_severity(df)
    df = drop_unused(df)
    df = fix_boolean_columns(df)

    return df

# 7. APPLY SMOTENC TO MODEL DATASET
# =====================================================
def smote_model_dataset(df):

    # MODEL = NO geographic columns
    regional_cols = ["State", "Start_Lat", "Start_Lng"]
    model_df = df.drop(columns=[c for c in regional_cols if c in df.columns], errors="ignore")

    # --- Train/test split ---
    y = model_df["Severity_Binary"]
    X = model_df.drop(columns=["Severity", "Severity_Binary"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y
    )

    # --- Handle missing values before SMOTE (required) ---
    for col in X_train.columns:
        if X_train[col].dtype == "object":
            # fill categorical with most common value
            mode = X_train[col].mode()[0]
            X_train[col] = X_train[col].fillna(mode)
            X_test[col] = X_test[col].fillna(mode)
        else:
            # fill numeric with train median
            median = X_train[col].median()
            X_train[col] = X_train[col].fillna(median)
            X_test[col] = X_test[col].fillna(median)

    # --- Identify categorical columns FOR SMOTENC ---
    categorical_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
    #SMOTENC requires the columns indices as integer values for NumPy
    categorical_indices = [X_train.columns.get_loc(col) for col in categorical_cols]

    #create a SMOTENC object that makes the minority class 50% of the majority (creating roughly 1.4 million points)
    #NOTE: SMOTENC must only be applied to NON ONE-HOT encoded features
    smote = SMOTENC(
        categorical_features=categorical_indices,
        random_state=None,
        sampling_strategy=0.5,
    )

    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # --- One-hot encode AFTER SMOTE (train + test)
    X_train_res = pd.get_dummies(X_train_res, drop_first=False)
    X_test = pd.get_dummies(X_test, drop_first=False)

    # Align columns
    X_train_res, X_test = X_train_res.align(X_test, join="left", axis=1, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    return (
        pd.DataFrame(X_train_scaled, columns=X_train_res.columns),
        pd.DataFrame(X_test_scaled, columns=X_test.columns),
        y_train_res.reset_index(drop=True),
        y_test.reset_index(drop=True)
    )


# 4. RUN SCRIPT
def main():
    df = pd.read_csv("Data/US_Accidents_Cleaned.csv")

    bad_mask = pd.to_datetime(df["Start_Time"], errors="coerce").isna()

    print("\n=== BAD ROW COUNT ===")
    print(bad_mask.sum())

    print("\n=== ACTUAL BAD Start_Time ROWS ===")
    print(df.loc[bad_mask, "Start_Time"].head(20))

    print("\n=== CHECKING NULL COUNTS BEFORE PREPROCESSING ===")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    # Check Start_Time status BEFORE any feature engineering
    print("\n=== Checking Start_Time validity BEFORE preprocessing ===")
    bad_st = pd.to_datetime(df["Start_Time"], errors="coerce").isna().sum()
    print("Invalid Start_Time rows:", bad_st)

    df = preprocess(df)

    print("\n=== CHECKING NULL COUNTS AFTER FEATURE ENGINEERING ===")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    # save regional dataset
    #df.to_parquet("Data/US_Accidents_Regional.parquet", index=False)

    # Build MODEL dataset
    X_train, X_test, y_train, y_test = smote_model_dataset(df)

    # Save X and y separately
    X_train.to_parquet("Data/US_Accidents_Model_X_Train.parquet", index=False, engine="pyarrow")
    X_test.to_parquet("Data/US_Accidents_Model_X_Test.parquet", index=False, engine="pyarrow")

    y_train = y_train.to_frame(name="Severity_Binary")
    y_train.to_parquet("Data/US_Accidents_Model_y_Train.parquet", index=False, engine="pyarrow")

    y_test = y_test.to_frame(name="Severity_Binary")
    y_test.to_parquet("Data/US_Accidents_Model_y_Test.parquet", index=False, engine="pyarrow")


if __name__ == "__main__":
    main()