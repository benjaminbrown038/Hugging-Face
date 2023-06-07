from datasets import load_dataset
import tensorflow as tf
import torch
import evaludate
import numpy as np
from PIL import Image
from tensorflow.keras import layers 
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from torchvision.transforms import RandomResizedCrop, Compose, Normal

from huggingface_hub import notebook_login
from transformers import AutoImageProcessor, DefaultDataCollator, AutoModelForImageClassification, TrainingArguments, Trainer, create_optimizer, TFAutoModelForImageClassification, pipeline
from datasets import load_dataset
from transformers.keras_callback import KerasMetricCallback, PushToCallback

food = load_dataset("food101", split = "train[:5000]")
food  = food.train_test_split(test_size = 0.2)
labels = food["train"].features["label"].names

label2id, id2label = dict(),dict()
for,label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    
id2label[str(79)]

checkpoint = "google/cit-base-pathc16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
normalize = Normalize(mean = image_processor.image_mean,std = image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_prcoessor.size["height"], image_processor.size["width"]
         )
  
  
  
transforms = Compose([RandomResizedCrop(size),ToTensor(),normalize])

def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for i in ]
    del example["image"]
    return examples
                   
'''
Tensorflow
'''
  
size = (image_processor.size["height"],image_processor.size["width"])
  
train_data_augmentation = keras.Sequential([layers.RandomCrop(size[0],size[1]),
                                            layers.Rescaling(scale = 1.0/127.5),
                                            layers.RandomFlip("horizontal"),
                                            layers.RandomRotation(factor=.02),
                                            layers.RandomZoom(height_factor=.2,width_factor=.2)
                                           ],
                                           name = "train_data_augmentation"
)
  
val_data_augmentation = keras.Sequential([layers.CenterCrop(size[0],size[1]),
                                          layers.Rescaling(scale = 1.0/127.5,offset = -1)
                                         ],
                                         name = "val_data_augmentation"
                                        )
  
  
def convert_to_tf_tensor():

def preprocess_train():

def preprocess_val():
 
def compute_metrics():

  
model = AutoModelForImageClassification.from_pretrained()
  
training_args = TrainingArguments()
  
trainer = Trainer()
  
trainer.train()
  
trainer.push_to_hub()

optimizer,lr_scheduler = create_optimizer()
  
model = TFAutoModelForImageClassification()
  
 
                                            
                                                     

          
          
          
          
          
          
