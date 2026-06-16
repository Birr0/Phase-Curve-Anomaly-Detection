import os
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from tqdm import tqdm
import light_curve as lc

load_dotenv()

HF_USERNAME = os.getenv("HF_USERNAME")

features = {
    #"Amplitude": lc.Amplitude(),
    #"AndersonDarlingNormal": lc.AndersonDarlingNormal(),
    "BeyondNStd": lc.BeyondNStd(nstd=2),
    "Cusum": lc.Cusum(),
    #"Eta": lc.Eta(),
    #"EtaE": lc.EtaE(),
    #"ExcessVariance": lc.ExcessVariance(),
    "InterPercentileRange": lc.InterPercentileRange(),
    "Kurtosis": lc.Kurtosis(),
    #"LinearFit": lc.LinearFit(),
    #"LinearTrend": lc.LinearTrend(),
    #"MagnitudePercentageRatio": lc.MagnitudePercentageRatio(),
    #"MaximumSlope": lc.MaximumSlope(),
    #"Mean": lc.Mean(),
    #"MeanVariance": lc.MeanVariance(),
    #"Median": lc.Median(),
    "MedianAbsoluteDeviation": lc.MedianAbsoluteDeviation(),
    #"MedianBufferRangePercentage": lc.MedianBufferRangePercentage(),
    #"OtsuSplit": lc.OtsuSplit(),
    #"PercentAmplitude": lc.PercentAmplitude(),
    #"PercentDifferenceMagnitudePercentile": lc.PercentDifferenceMagnitudePercentile(),
    "ReducedChi2": lc.ReducedChi2(),
    #"Roms": lc.Roms(),
    "Skew": lc.Skew(),
    "StandardDeviation": lc.StandardDeviation(),
    #"StetsonK": lc.StetsonK(),
    #"WeightedMean": lc.WeightedMean(),
}

#ReducedChi2, Kurtosis, Skew, StetsonK, BeyondNStd, MaximumSlope, InterPercentileRange, MedianAbsoluteDeviation, StandardDeviation, and Cusum

data_files = {"full": "./sso_fink_ztf_lc.parquet"}

print("Preparing to load dataset")
ds = load_dataset("parquet", data_files=data_files)["full"]

print("Loaded dataset.")

# Load columns into memory once
residuals_list = ds["residuals_shg1g2"]
errors_list = ds["csigmapsf"]
phases_list = ds["Phase"]
cfid_list = ds["cfid"]
ssnamenr_list = ds["ssnamenr"]
bands = ds["cfid"]

for i in range(10):
    mask = bands[0] == 1
    cjd_array = np.array(ds["cjd"][i])[mask]

    is_ascending = np.all(np.diff(cjd_array) >= 0)
    print(f"Is ascending: {is_ascending}")

    # Check if it is strictly ascending (no duplicate timestamps)
    is_strictly_ascending = np.all(np.diff(cjd_array) > 0)
    print(f"Is strictly ascending: {is_strictly_ascending}")

extractor = lc.Extractor(*list(features.values()))
feature_names = extractor.names

rows_g = []
rows_r = []
sso_number_fails = []

def build_band_row(phase, resid, err, ssnamenr, band_suffix):
    """
    Returns a feature row dict, or None if the band should be dropped.
    Drops bands with <10 observations or any NaN/inf in inputs/features.
    """
    valid = np.isfinite(phase) & np.isfinite(resid) & np.isfinite(err)
    phase = phase[valid]
    resid = resid[valid]
    err = err[valid]

    if len(resid) < 10:
        return None

    #idx = np.argsort(phase)
    #phase = phase[idx]
    #resid = resid[idx]
    #err = err[idx]

    try:
        result = extractor(phase, resid, err, sorted=True, check=False)
    except Exception:
        return None

    result = np.asarray(result)

    # Drop any row with NaN / inf in the extracted features
    if not np.all(np.isfinite(result)):
        return None

    row = {f"{name}_{band_suffix}": value for name, value in zip(feature_names, result)}
    row["ssnamenr"] = ssnamenr
    return row


print("Processing light curves...")
for resid, err, phase, bands, ssnamenr in tqdm(
    zip(residuals_list, errors_list, phases_list, cfid_list, ssnamenr_list),
    total=len(ds),
    desc="Processing light curves",
):
    resid = np.asarray(resid)
    err = np.asarray(err)
    phase = np.asarray(phase)
    bands = np.asarray(bands)

    if len(resid) == 0:
        sso_number_fails.append(ssnamenr)
        continue

    # Basic finite-mask for the full light curve
    finite = np.isfinite(resid) & np.isfinite(err) & np.isfinite(phase)
    resid = resid[finite]
    err = err[finite]
    phase = phase[finite]
    bands = bands[finite]
    print()
    if len(resid) == 0:
        sso_number_fails.append(ssnamenr)
        continue

    # g band = 1
    g_mask = bands == 1
    row_g = build_band_row(
        phase[g_mask], resid[g_mask], err[g_mask], ssnamenr, "g"
    )
    if row_g is not None:
        rows_g.append(row_g)

    # r band = 2
    r_mask = bands == 2
    row_r = build_band_row(
        phase[r_mask], resid[r_mask], err[r_mask], ssnamenr, "r"
    )
    if row_r is not None:
        rows_r.append(row_r)

print("Converting to DataFrames...")

df_g = pd.DataFrame(rows_g).dropna()
df_r = pd.DataFrame(rows_r).dropna()

print(f"g-band rows kept: {len(df_g)}")
print(f"r-band rows kept: {len(df_r)}")

# Save separately
df_g.to_parquet("./residual_features_g.parquet", index=False)
df_r.to_parquet("./residual_features_r.parquet", index=False)

# Save failures
fails_df = pd.DataFrame({"ssnamenr": sso_number_fails})
fails_df.to_csv("./failed_sso_numbers.csv", index=False)

# Push separately to Hub
Dataset.from_pandas(df_g).push_to_hub(
    f"{HF_USERNAME}/residual_features_g",
    private=False,
)

Dataset.from_pandas(df_r).push_to_hub(
    f"{HF_USERNAME}/residual_features_r",
    private=False,
)

print("Finished!")

'''
import os 

import light_curve as lc
import numpy as np
import pandas as pd 
from datasets import load_dataset 
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv() 

HF_USERNAME = os.getenv("HF_USERNAME")

features = {
    "Amplitude": lc.Amplitude(),
    "AndersonDarlingNormal": lc.AndersonDarlingNormal(),
    "BeyondNStd": lc.BeyondNStd(nstd=1),
    "Cusum": lc.Cusum(),
    "Eta": lc.Eta(),
    "EtaE": lc.EtaE(),
    "ExcessVariance": lc.ExcessVariance(),
    "InterPercentileRange": lc.InterPercentileRange(),
    "Kurtosis": lc.Kurtosis(),
    "LinearFit": lc.LinearFit(),
    "LinearTrend": lc.LinearTrend(),
    "MagnitudePercentageRatio": lc.MagnitudePercentageRatio(),
    "MaximumSlope": lc.MaximumSlope(),
    "Mean": lc.Mean(),
    "MeanVariance": lc.MeanVariance(),
    "Median": lc.Median(),
    "MedianAbsoluteDeviation": lc.MedianAbsoluteDeviation(),
    "MedianBufferRangePercentage": lc.MedianBufferRangePercentage(),
    "OtsuSplit": lc.OtsuSplit(),
    "PercentAmplitude": lc.PercentAmplitude(),
    "PercentDifferenceMagnitudePercentile": lc.PercentDifferenceMagnitudePercentile(),
    "ReducedChi2": lc.ReducedChi2(),
    "Roms": lc.Roms(),
    "Skew": lc.Skew(),
    "StandardDeviation": lc.StandardDeviation(),
    "StetsonK": lc.StetsonK(),
    "WeightedMean": lc.WeightedMean(),
}

data_files = {
    "full" : f"./sso_fink_ztf_lc.parquet",
}
print("Preparing to load dataset")
ds = load_dataset("parquet", data_files=data_files)["full"]
rows = []
sso_number_fails = []

print("Loaded dataset. Extracting columns into memory for faster processing...")

# ---- SPEED-UP 1: Load columns into memory once ----
residuals_list = ds["residuals_shg1g2"]
errors_list = ds["csigmapsf"]
phases_list = ds["Phase"]
cfid_list = ds["cfid"]
ssnamenr_list = ds["ssnamenr"]

# ---- SPEED-UP 2: Instantiate extractor ONCE outside the loop ----
extractor = lc.Extractor(*list(features.values()))

# ---- SPEED-UP 3: Zip the in-memory lists together ----
for resid, err, phase, bands, ssnamenr in tqdm(
    zip(residuals_list, errors_list, phases_list, cfid_list, ssnamenr_list), 
    total=len(ds), 
    desc="Processing light curves"
):
    resid = np.array(resid)
    err = np.array(err)
    phase = np.array(phase)
    
    if len(resid) == 0:
        sso_number_fails.append(ssnamenr)
        continue

    idx = np.argsort(phase)
    phase = phase[idx]
    resid = resid[idx]
    err  = err[idx]
    
    # Calculate joint features
    result_joint = extractor(phase, resid, err, sorted=True, check=False)

    bands = np.array(bands)[idx]
    g_mask = bands == 1
    r_mask = bands == 2

    # Calculate G band features
    resid_g = resid[g_mask]
    err_g = err[g_mask]
    phase_g = phase[g_mask]

    try:
        result_g = extractor(phase_g, resid_g, err_g, sorted=True, check=False)
    except Exception as e:
        # FIX: Use len(extractor.names) instead of len(features)
        result_g = [np.nan] * len(extractor.names) 
   
    # Calculate R band features
    resid_r = resid[r_mask]
    err_r = err[r_mask]
    phase_r = phase[r_mask]

    try:
        result_r = extractor(phase_r, resid_r, err_r, sorted=True, check=False)
    except Exception as e:
        # FIX: Use len(extractor.names) instead of len(features)
        result_r = [np.nan] * len(extractor.names)

    # ---- build row dictionary ----
    row = {}

    # joint features
    for name, value in zip(extractor.names, result_joint):
        row[f"{name}_joint"] = value

    # g features
    for name, value in zip(extractor.names, result_g):
        row[f"{name}_g"] = value

    # r features
    for name, value in zip(extractor.names, result_r):
        row[f"{name}_r"] = value
        
    row["ssnamenr"] = ssnamenr
    rows.append(row)

# ---- Convert to DataFrame ----
print("Converting to DataFrame and saving...")
df = pd.DataFrame(rows)

# ---- Save to parquet ----
df.to_parquet("./residual_features.parquet", index=False)
fails_df = pd.DataFrame({"ssnamenr": sso_number_fails})
fails_df.to_csv("failed_sso_numbers.csv", index=False)

# ---- Push to Hub ----
data_files_out = {
    "full" : f"./residual_features.parquet",
}
ds_dict = load_dataset("parquet", data_files=data_files_out)
ds_dict.push_to_hub(
    f"{HF_USERNAME}/residual_features",
    private=False,     
)
print("Finished!")
'''

'''
import os 

import light_curve as lc
import numpy as np
from datasets import load_dataset
import pandas as pd 
from datasets import load_dataset 
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv() 

HF_USERNAME = os.getenv("HF_USERNAME")

features = {
    "Amplitude": lc.Amplitude(),
    "AndersonDarlingNormal": lc.AndersonDarlingNormal(),
    "BeyondNStd": lc.BeyondNStd(nstd=1),
    "Cusum": lc.Cusum(),
    "Eta": lc.Eta(),
    "EtaE": lc.EtaE(),
    "ExcessVariance": lc.ExcessVariance(),
    "InterPercentileRange": lc.InterPercentileRange(),
    "Kurtosis": lc.Kurtosis(),
    "LinearFit": lc.LinearFit(),
    "LinearTrend": lc.LinearTrend(),
    "MagnitudePercentageRatio": lc.MagnitudePercentageRatio(),
    "MaximumSlope": lc.MaximumSlope(),
    "Mean": lc.Mean(),
    "MeanVariance": lc.MeanVariance(),
    "Median": lc.Median(),
    "MedianAbsoluteDeviation": lc.MedianAbsoluteDeviation(),
    "MedianBufferRangePercentage": lc.MedianBufferRangePercentage(),
    "OtsuSplit": lc.OtsuSplit(),
    "PercentAmplitude": lc.PercentAmplitude(),
    "PercentDifferenceMagnitudePercentile": lc.PercentDifferenceMagnitudePercentile(),
    "ReducedChi2": lc.ReducedChi2(),
    "Roms": lc.Roms(),
    "Skew": lc.Skew(),
    "StandardDeviation": lc.StandardDeviation(),
    "StetsonK": lc.StetsonK(),
    "WeightedMean": lc.WeightedMean(),
}

data_files = {
    "full" : f"./sso_fink_ztf_lc.parquet",
}
print("Preparing to load dataset")
ds = load_dataset("parquet", data_files=data_files)["full"] #.with_format("numpy")
rows = []

sso_number_fails = []


print("Loaded dataset")

for i in tqdm(range(len(ds)), desc="Processing light curves"):
    resid = np.array(ds["residuals_shg1g2"][i])
    err = np.array(ds["csigmapsf"][i])
    phase = np.array(ds["Phase"][i])
    
    if len(resid) == 0:
        sso_number_fails.append(
            ds["ssnamenr"][i]
        )
        continue

    idx = np.argsort(phase)
    phase = phase[idx]
    resid = resid[idx]
    err  = err[idx]
    
    extractor = lc.Extractor(*list(features.values()))
    
    result_joint = extractor(phase, resid, err, sorted=True, check=False)

    bands = np.array(ds["cfid"][i])
    bands = bands[idx]
    g_mask = bands == 1
    r_mask = bands == 2

    resid_g = resid[g_mask]
    err_g = err[g_mask]
    phase_g = phase[g_mask]

    try:
        result_g = extractor(phase_g, resid_g, err_g, sorted=True, check=False)
    except Exception as e:
        # fallback to NaNs if extractor fails unexpectedly
        result_g = [np.nan] * len(features)
   
    resid_r = resid[r_mask]
    err_r = err[r_mask]
    phase_r = phase[r_mask]
    result_r = extractor(phase_r, resid_r, err_r, sorted=True, check=False)

    try:
        result_r = extractor(phase_r, resid_r, err_r, sorted=True, check=False)
    except Exception as e:
        # fallback to NaNs if extractor fails unexpectedly
        result_r = [np.nan] * len(features)

    # ---- build row dictionary ----
    row = {}

    # joint features
    for name, value in zip(extractor.names, result_joint):
        row[f"{name}_joint"] = value

    # g features
    for name, value in zip(extractor.names, result_g):
        row[f"{name}_g"] = value

    # r features
    for name, value in zip(extractor.names, result_r):
        row[f"{name}_r"] = value
    row["ssnamenr"] = ds["ssnamenr"][i]
    rows.append(row)

# ---- Convert to DataFrame ----
df = pd.DataFrame(rows)
# ---- Save to parquet ----
df.to_parquet("./residual_features.parquet", index=False)
fails_df = pd.DataFrame({"ssnamenr": sso_number_fails})
fails_df.to_csv("failed_sso_numbers.csv", index=False)

data_files = {
    "full" : f"./residual_features.parquet",
}
ds_dict = load_dataset("parquet", data_files=data_files)
ds_dict.push_to_hub(
    f"{HF_USERNAME}/residual_features",
    private=False,     
)
'''

'''
lc_features = {
    "Median": lc.Median(), # median in each band
    "BazinFit": lc.BazinFit('mcmc'),
    "RainbowFit": lc.RainbowFit(), # not super relevant
    "VillarFit": lc.VillarFit(), #
    "FluxNNotDetBeforeFd": lc.FluxNNotDetBeforeFd(),
}
'''