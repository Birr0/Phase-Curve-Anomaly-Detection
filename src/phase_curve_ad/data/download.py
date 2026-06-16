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
  "https://api.ztf.fink-portal.org/api/v1/ssobulk",
  json={
    "output-format": "parquet"
  }
)

with open("./sso_fink_ztf_lc.parquet", "wb") as f:
    f.write(r.content)

data_files = {
    "full" : f"./sso_fink_ztf_lc.parquet",
}
ds_dict = load_dataset("parquet", data_files=data_files)
ds_dict.push_to_hub(
    f"{HF_USERNAME}/ztf_sso_bulk",
    private=False,     
)