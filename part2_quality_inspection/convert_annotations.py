"""
Convert DeepPCB annotations to YOLO format for quality inspection task
Maps original classes to: missing_component, misaligned_component, solder_defect
"""

import os
import glob
from PIL import Image

# Source and destination paths
source_img_dir = "/home/ziva/ai_vision_assignment/part2_quality_inspection/data/images/DeepPCB"
source_anno_dir = "/home/ziva/Downloads/DeepPCB-master/PCBData"
dest_anno_dir = "/home/ziva/ai_vision_assignment/part2_quality_inspection/data/annotations"

# DeepPCB has 6 defect classes, we'll map them to our 3 classes
# Based on typical PCB defects:
# class 1,2 -> missing_component (0)
# class 3,4 -> misaligned_component (1)  
# class 5,6 -> solder_defect (2)

class_mapping = {
    1: 0,  # missing_component
    2: 0,  # missing_component
    3: 1,  # misaligned_component
    4: 1,  # misaligned_component
    5: 2,  # solder_defect
    6: 2   # solder_defect
}

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert bounding box from (x1, y1, x2, y2) to YOLO format (x_center, y_center, width, height)
    All values normalized to [0, 1]
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate center and dimensions
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return x_center, y_center, width, height

def find_annotation_file(img_name):
    """Find the corresponding annotation file for an image"""
    # Extract base name (e.g., 12300060 from 12300060_test.jpg)
    base_name = img_name.replace("_test.jpg", "").replace("_temp.jpg", "")
    
    # Search in DeepPCB folders - try multiple patterns
    search_patterns = [
        f"{source_anno_dir}/**/*_not/{base_name}.txt",
        f"{source_anno_dir}/**/not/{base_name}.txt",
        f"{source_anno_dir}/**/{base_name}.txt"
    ]
    
    for pattern in search_patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            return files[0]
    
    return None

def convert_annotations():
    """Convert all annotations from DeepPCB format to YOLO format"""
    
    os.makedirs(dest_anno_dir, exist_ok=True)
    
    # Get all images
    image_files = glob.glob(os.path.join(source_img_dir, "*.jpg"))
    
    converted_count = 0
    clean_images = 0
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        
        # Find corresponding annotation
        anno_file = find_annotation_file(img_name)
        
        if not anno_file or not os.path.exists(anno_file):
            # No defects for this image (clean PCB)
            clean_images += 1
            continue
        
        # Get image dimensions
        try:
            img = Image.open(img_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening {img_name}: {e}")
            continue
        
        # Read original annotations
        yolo_annotations = []
        
        with open(anno_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                x1, y1, x2, y2, orig_class = map(int, parts[:5])
                
                # Map to our 3 classes
                if orig_class in class_mapping:
                    new_class = class_mapping[orig_class]
                    
                    # Convert to YOLO format
                    x_center, y_center, width, height = convert_bbox_to_yolo(
                        (x1, y1, x2, y2), img_width, img_height
                    )
                    
                    yolo_annotations.append(
                        f"{new_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )
        
        # Save YOLO format annotation
        if yolo_annotations:
            output_file = os.path.join(dest_anno_dir, img_name.replace('.jpg', '.txt'))
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            converted_count += 1
    
    print(f"\n✅ Conversion Complete!")
    print(f"   Converted: {converted_count} images with defects")
    print(f"   Clean PCBs: {clean_images} images")
    print(f"   Total: {len(image_files)} images")
    print(f"\n📁 Annotations saved to: {dest_anno_dir}")

if __name__ == "__main__":
    convert_annotations()
