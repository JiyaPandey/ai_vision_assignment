import os
import shutil

# Source (original COCO mini dataset)
SRC_IMAGES = "coco128/images/train2017"
SRC_LABELS = "coco128/labels/train2017"

# Destination (our custom 5-class dataset)
DST_IMAGES = "coco5/images/train"
DST_LABELS = "coco5/labels/train"

# Keep only these COCO classes and remap them to 0–4
# person=0, car=1, dog=2, bottle=3, chair=4
KEEP_CLASSES = {
    0: 0,    # person
    2: 1,    # car
    16: 2,   # dog
    39: 3,   # bottle
    56: 4    # chair
}

# Create destination folders
os.makedirs(DST_IMAGES, exist_ok=True)
os.makedirs(DST_LABELS, exist_ok=True)

kept_images = 0

for label_file in os.listdir(SRC_LABELS):
    src_label_path = os.path.join(SRC_LABELS, label_file)

    with open(src_label_path, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        parts = line.strip().split()
        cls_id = int(parts[0])

        if cls_id in KEEP_CLASSES:
            parts[0] = str(KEEP_CLASSES[cls_id])  # remap class id
            new_lines.append(" ".join(parts))

    if new_lines:
        img_name = label_file.replace(".txt", ".jpg")
        src_img_path = os.path.join(SRC_IMAGES, img_name)

        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, os.path.join(DST_IMAGES, img_name))
            with open(os.path.join(DST_LABELS, label_file), "w") as f:
                f.write("\n".join(new_lines))
            kept_images += 1

print("✅ coco5 dataset created")
print("Images kept:", kept_images)
