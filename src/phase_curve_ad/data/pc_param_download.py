import os
import io

import requests
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login()

HF_USERNAME = os.getenv("HF_USERNAME")

r = requests.post(
  "https://api.ztf.fink-portal.org/api/v1/ssoft",
  json={
    "flavor":   "SHG1G2",
    "version": "2023.12",
    "output-format": "parquet"
  }
)
# 1. ADD THIS: Catch API errors immediately
if r.status_code != 200:
    raise RuntimeError(f"API Request failed! Status: {r.status_code}\nMessage: {r.text}")

print("Writing to parquet file...")
with open("pc_params.parquet", "wb") as f:
    f.write(r.content)

data_files = {
    "full" : f"./*.parquet",
}

ds_dict = load_dataset("parquet", data_files=data_files)
ds_dict.push_to_hub(
    f"{HF_USERNAME}/sHG1G2_params",
    private=False,     
)