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


def train_transform(example_batch):
    images = [jitter(x) for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images,labels)
    return inputs

def val_transforms():
    images = [x for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images,labels)
    return inputs
  
def aug_tranforms():
    image = tf.keras.utils.image_to_array(image)
    image = tf.image.random.brightness(image,0.25)
    image = tf.image.random_contrast(image,0.5,2.0)
    image = tf.image.random_saturation(image,0.75,1.25)
    image = tf.image.random_hue(image,0.1)
    image = tf.transpose(image,(2,0,1))
    return image

def transforms():
    image = tf.keras.utils.img_to_array(image)
    image = tf.transpose(image,(2,0,1))
    return image

def train_transform():
    images = [aug_transforms(x.convert("RGB")) for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images,labels)
    return inputs
  
def val_transforms():
    images = [transforms(x.convert("RGB")) for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images,labels)
    return inputs
 
def compute_metrics():
    with torch.no_grad():
        logits.labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size = labels.shape[-2:1],
            mode = "bilinear",
            align_corners = False).argmax(dim = 1)
        ).argmax(dim=1)
        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions = pred_labels,
            references = labels,
            num_labels = num_labels,
            ignore_index = 255,
            reduce_labels = False)
        return metrics
          
model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint,id2label = id2label, label2id = label2id)

training_args = TrainingArguments(
  output_dir,
  learning_rate,
  num_train,
  per_device_train_batch_size,
  per_device_eval_batch_size,
  save_total_limit,
  evaluation_strategy,
  save_strategy,
  save_strategy,
  save_steps, 
  eval_steps,
  logging_steps,
  eval_accumulation_steps,
  remove_unused_columns,
  push_to_hub)

trainer = Trainer(
  model = model,
  args = training_args,
  train_dataset = train_ds,
  eval_dataset = test_ds,
  compute_metrics = compute_metrics
)

trainer.train()
train.push_to_hub()

batch_size = 2
num_epochs = 50 
num_train_steps = len(train_ds) * num_epochs
learning_rate = 6e-5
weight_decay_rate = .01
optimizer, lr_schedule = create_optimizer(
  init_lr = learning_rate,
  num_train_steps = num_train_steps,
  weight_decay_rate = weight_decay_rate,
  num_warmup_steps = 0)

model = TFAutoModelForSemanticSegmentation.from_pretrained(
    checkpoint,
  id2label=id2label,
  label2id=label2id)
model.compile(optimizer = optimizer)
data_collator = DefaultDataCollator(return_tensors = "tf")

tf_train_dataset = train_ds.to_tf_dataset(
    columns = ["pixel_values","label"],
    shuffle = True,
    batch_size = batch_size,
    collate_fn = data_collator)

tf_eval_dataset = test_ds.to_tf_dataset(
    columns = ["pixel_values","label"],
    shuffle = True,
    batch_size = batch_size,
    collate_fn = data_collator)

tf_eval_dataset = test_ds.to_tf_dataset(
    columns = ["pixel_values","label"],
    shuffle = True, 
    batch_size = batch_size,
    collate_fn = data_collator)

metric_callback = KerasMetricCallback(
    metric_fn = compute_metrics,
    eval_dataset = tf_eval_dataset,
    batch_size = batch_size,
    label_cols = ["labels"])










  
  
  
  
  
  
  
  
  
  
  
  
  
  
