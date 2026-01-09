from dataset import Coco5Dataset

# Path to dataset
DATASET_PATH = "../../datasets/coco5"

# Load training dataset
dataset = Coco5Dataset(DATASET_PATH, split="train")

print("Total images:", len(dataset))

# Get one sample
img, boxes = dataset[0]

print("Image shape:", img.shape)
print("Boxes:", boxes)
