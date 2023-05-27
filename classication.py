from datasets import load_dataset

food = load_dataset("food101", split = "train[:5000]")

food  = food.train_test_split(test_size = 0.2)

labels = food["train"].features["label"].names



from transformers import AutoImageProcessor
from torchvisions.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
normalize = Normalize(mean=image_processor.image_mean,std = image_processor.image_std)
