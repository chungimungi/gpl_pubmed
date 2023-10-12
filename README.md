# Generative Pseudo Labelling

## Download all required packages using:

`pip install -r requirements.txt`



```python
from huggingface_hub import hf_hub_download
import zipfile
from datasets import load_dataset

#load the dataset
dataset_id = "chungimungi/pubmed"
filename = "pubmed.zip"
hf_hub_download(repo_id=dataset_id, filename=filename)

# Unzip the dataset
with zipfile.ZipFile(filename, "r") as zip_ref:
    zip_ref.extractall()

# Load the dataset
dataset = load_dataset("pubmed")
```
