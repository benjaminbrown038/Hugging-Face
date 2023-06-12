from huggingface_hub import notebook_login
import numpy as np
from datasets import load_dataset
import os
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor

cppe5 = load_dataset("cppe-5")
cppe5["train"][0]

image = cppe5["train"][0]["image"]
annotations = cppe5["train"][0]["objects"]
draw = ImageDraw.Draw(image)

categories = cppe5["train"].features["objects"].feature["category"].names
id2label = {index: x for index,x in enumerate(categories,start=0)}
label2id = {v: k for k,v in id2label.items()}
for i in range(len(annotations["id"])):
    box = annotations["bbox"][i-1]
    class_idx = annotations["category"][i-1]
    x,y,w,h = tuple(box)
    draw.rectangle((x,y,x+w,y+h), outline = "red", width = 1)
    draw.text((x,y),id2label[class_idx],fill = "white")

keep = [i for i in range(len(cppe5["train"])) if i not in remove_idx]
cppe5["train"] = cppe5["train"].select(keep)

checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

transform = albumentations.Compose(
      [
          albumentations.Resize(480,480),
          albumentations.HorizontalFlip(p=1.0)
          albumentations.RandomBrightnessContrast(p=1.0)
      ],
      bbox_params=albumentations.BboxParams(format = "coco",label_fields = ["category"])
)

def formatted_anns(image_id,category,area,bbox):
    annotations = []
    for i in range(0,len(category)):
        new_ann = {
            "image_id":image_id,
            "category_id":category[i],
            "isCrowd":0,
            "area":area[i],
            "bbox": list(bbox[i])
        }
        annotations.append(new_ann)
      return annotations
    
def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images,bboxes,area,category = [],[],[],[]
    for image, objects in zip(examples["images"],examples["objects"]):
        image = np.array(image.convert("RGB"))[:,:,::-1]
        out = transform(image = image,bboxes=objects["bbox"],category=objects["category"])

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
      {"image_id":id_,"annotations":formatted_anns(id_,cat_,ar_,box_)}
      for id_,cat_,ar_,box_ in zip(image_ids,categories,area,bboxes)
    ]
    return image_processor(images=images,annotations = targets, return_tensors = "pt")
   
cppe5["train"] = cppe["train"].with_transform(transform_aug_ann)
cppe["train"][15]

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad_and_create_pixel_mask(pixel_values,return_tensors = "pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch
model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label,
    label2id=label2id,
    ignore_mismatched_sizes = True
)

training_args = TrainingArguments(
    output_dir = "detr-resnet-50_finetuned_cppe5",
    per_device_train_batch_size = 8,
    num_train_epochs = 8,
    fp16 = True,
    save_steps = 200,
    logging_steps = 50,
    learning_rate = 1e-5,
    weight_decay = 1e-4,
    save_total_limit = 2,
    remove_unused_columns = False,
    push_to_hub = True
)

trainer = Trainer(
    model = model,
    args = training_args,
    data_collator=collate_fn,
    train_dataset = cppe5["train"]
    tokenizer = image_processor
)

trainer.train()
trainer.push_to_hub()

def val_formatted_anns(image_id,objects):
    annotations = []
    for i in range(0,len(objects["id"])):
        new_ann = {
            "id":objects["id"][i]
            "category_id": objects["category"][i]
            "iscrowd": 0,
            "image_id":image_id,
            "area": objects["area"][i],
            "bbox": objects["bbox"][i]
        }
    return annotations
  
def save_cppe5_annotation_file_image(cppe5):
    output_json = []
    path_output_cppe5 = f"{os.getcwd()}/cppe5"
    if not os.path.exists(path_output_cppe5):
        os.makedirs(path_output_cppe5)
    
    path_anno = os.path.join(path_output_cppe5,"cppe5_ann.json")
    categories_json = [{"supercategory":"none","id",id,"name",id2label[id]} for id in id2label]
    ouput_json["images"] = []
    output_json["annotations"] = []
    for example in cppe5:
        ann = val_formatted_anns(example["image_id",example["objects"]])
        output_json["images"].append(
            {
                "id": example["image_id"],
                "width": example["image"].width,
                "height":example["image"].height,
                "file_name": f"{example['image_id']}.png"
            }
        )
        output_json["annotations"].extend(ann)
    output_json["categories"] = categories_json

    with open(path_anno,"w") as file:
        json.dump(output_json,file,ensure_ascii=False,endent=4)
    
    for im, img_id in zip(cppe5["image"],cppe5["image_id"]):
        path_img = os.path.join(path_output_cppe5,f"{img_id}.png")
        im.save(path_img)
        
    return path_output_cppe5, pathh_anno
  
  
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self,img_folder,feature_extractor, ann_file):
        super().__init__(img_folder,ann_file)
        self.feature_extractor = feature_extractor
    
    def __getitem__(self,idx):
        img,target = super(CocoDetection,self).__getitem__(idx)
        image_id = self.ids






