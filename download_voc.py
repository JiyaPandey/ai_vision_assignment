from torchvision.datasets import VOCDetection

dataset = VOCDetection(
    root="part1_object_detection/data",
    year="2012",
    image_set="trainval",
    download=True
)

print("VOC 2012 downloaded successfully!")
print("Number of images:", len(dataset))
