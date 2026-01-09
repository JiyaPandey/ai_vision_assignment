"""
Create train/val split for Pascal VOC 2007 with 5 classes
"""
import os
import xml.etree.ElementTree as ET
import random

VOC_ROOT = os.path.expanduser("~/ai_vision_assignment/datasets/VOCdevkit/VOC2007")
TARGET_CLASSES = ["person", "car", "dog", "cat", "bicycle"]
TRAIN_RATIO = 0.8

def contains_target_class(ann_path):
    """Check if annotation contains any target class"""
    try:
        tree = ET.parse(ann_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            if obj.find('name').text in TARGET_CLASSES:
                return True
        return False
    except:
        return False

# Get all image IDs from all class files
main_dir = os.path.join(VOC_ROOT, "ImageSets/Main")
all_ids = set()

for class_name in TARGET_CLASSES:
    trainval_file = os.path.join(main_dir, f"{class_name}_trainval.txt")
    with open(trainval_file, 'r') as f:
        for line in f:
            img_id = line.strip().split()[0]  # Format: "img_id 1/-1"
            all_ids.add(img_id)

all_ids = list(all_ids)

# Filter for images containing target classes
ann_dir = os.path.join(VOC_ROOT, "Annotations")
filtered_ids = []

print(f"Filtering {len(all_ids)} images for 5 classes...")
for img_id in all_ids:
    ann_path = os.path.join(ann_dir, f"{img_id}.xml")
    if contains_target_class(ann_path):
        filtered_ids.append(img_id)

print(f"Found {len(filtered_ids)} images with target classes")

# Randomly sample ~600 images if we have more
if len(filtered_ids) > 600:
    random.seed(42)
    filtered_ids = random.sample(filtered_ids, 600)
    print(f"Randomly selected 600 images")

# Shuffle and split
random.shuffle(filtered_ids)
split_idx = int(len(filtered_ids) * TRAIN_RATIO)
train_ids = filtered_ids[:split_idx]
val_ids = filtered_ids[split_idx:]

# Write to files
main_dir = os.path.join(VOC_ROOT, "ImageSets/Main")
with open(os.path.join(main_dir, "train.txt"), 'w') as f:
    f.write('\n'.join(train_ids))

with open(os.path.join(main_dir, "val.txt"), 'w') as f:
    f.write('\n'.join(val_ids))

print(f"\n✅ Split created:")
print(f"   Train: {len(train_ids)} images")
print(f"   Val: {len(val_ids)} images")
print(f"   Total: {len(filtered_ids)} images")
