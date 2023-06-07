import json
from huggingface_hub import notebook_login
from datasets import load_dataset
from huggingface_hub import cached_download, hf_hub_url
from transformers import AutoImageProcessor
from torchvision.transforms import ColorJitter
import tensorflow as tf
import evaluate 
from transformers import AutoModelForSemanticSegmentation, TrainingArguments
import matplotlib.pyplot as plt 
import numpy as np 



ds = load_dataset("scene_parse_150",split = "train[:50]")
ds = ds.train_test_split(test_size = 0.2)
train_ds = ds["train"]
test_ds = ds["test"]
train_ds[0]

repo_id = "huggingface/label-files"
filename = "ade20k-hf-doc.json"
id2label = json.load(open(cached_download(hf_hub_url(repo_id,filename))))
id2label = {int(k): v for k,v in id2label.items()}
label2id = {v: k for k,v in id2label.items()}
num_labels = len(id2label)


def train_transform():

def val_transforms():
  
def aug_tranforms():
  
def transforms():
  
def train_transforms():
  
def val_transforms():
  
def compute_metrics():
  
