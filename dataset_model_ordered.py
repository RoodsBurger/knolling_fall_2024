import os
import shutil
from pathlib import Path
import random
from collections import defaultdict
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

class SimpleObjectDetector:
    def __init__(self, confidence_threshold=0.2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        self.confidence_threshold = confidence_threshold

    def detect_objects(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(image_tensor)[0]

        boxes = []
        scores = []
        img_width, img_height = image.size

        for box, score in zip(predictions['boxes'], predictions['scores']):
            if score > self.confidence_threshold:
                x1, y1, x2, y2 = box.cpu().numpy()
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                boxes.append([x_center, y_center, width, height])
                scores.append(score.item())

        return boxes

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes in YOLO format."""
    # Convert from YOLO format (center_x, center_y, width, height) to corners
    def get_corners(box):
        center_x, center_y, width, height = box
        x1 = center_x - width/2
        y1 = center_y - height/2
        x2 = center_x + width/2
        y2 = center_y + height/2
        return x1, y1, x2, y2

    x1_1, y1_1, x2_1, y2_1 = get_corners(box1)
    x1_2, y1_2, x2_2, y2_2 = get_corners(box2)

    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

def filter_overlapping_boxes(boxes, overlap_threshold=0.1):
    """Filter out overlapping boxes, keeping the ones that appear higher in the image."""
    if not boxes:
        return []

    # Sort boxes by y-coordinate (top to bottom)
    boxes_with_index = [(i, box) for i, box in enumerate(boxes)]
    boxes_with_index.sort(key=lambda x: x[1][1])  # Sort by y-coordinate

    filtered_indices = []
    filtered_boxes = []

    for i, box1 in boxes_with_index:
        # Check if this box overlaps with any previously kept box
        overlaps = False
        for kept_box in filtered_boxes:
            if calculate_iou(box1, kept_box) > overlap_threshold:
                overlaps = True
                break
        
        if not overlaps:
            filtered_indices.append(i)
            filtered_boxes.append(box1)

    return filtered_boxes

def match_objects_to_labels(detected_boxes, class_names, pos_values, class_to_id, max_distance=2):
    if not detected_boxes or not class_names:
        return []

    # First, filter out overlapping detections
    filtered_boxes = filter_overlapping_boxes(detected_boxes)

    # If the number of filtered boxes doesn't match the number of classes, return empty
    if len(filtered_boxes) != len(class_names):
        return []

    MAX_X, MAX_Y = 0.3, 0.25
    original_positions = []
    
    # Get original positions from pos_values
    for i in range(len(class_names)):
        pos_idx = i * 7
        if pos_idx + 6 >= len(pos_values):
            return []

        y, x = pos_values[pos_idx:pos_idx + 2]
        v5, v6, v7 = pos_values[pos_idx + 4:pos_idx + 7]

        x_norm = x / MAX_X
        y_norm = y / MAX_Y

        original_positions.append({
            'coords': [x_norm, y_norm],
            'index': i,
            'extra': [v5, v6, v7]
        })

    # Sort both arrays strictly by y first, then x for equal y values
    original_positions.sort(key=lambda p: (round(p['coords'][1], 3), p['coords'][0]))
    
    detected_boxes_info = []
    for i, box in enumerate(filtered_boxes):
        detected_boxes_info.append({
            'coords': [box[0], box[1]],
            'index': i,
            'box': box
        })
    detected_boxes_info.sort(key=lambda p: (round(p['coords'][1], 3), p['coords'][0]))

    # One-to-one matching based on sorted order
    new_annotations = []
    for orig, det in zip(original_positions, detected_boxes_info):
        # Check if the distance between matched points is within threshold
        dist = ((orig['coords'][0] - det['coords'][0]) ** 2 + 
                (orig['coords'][1] - det['coords'][1]) ** 2) ** 0.5
        
        if dist > max_distance:
            return []
            
        class_id = class_to_id[class_names[orig['index']]]
        new_annotations.append([class_id, *det['box']])

    return new_annotations

def create_yolo_dataset(base_dir, output_dir, train_ratio=0.8, max_batch=2000):
    detector = SimpleObjectDetector(confidence_threshold=0.2)

    yolo_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for dir_path in yolo_dirs:
        Path(os.path.join(output_dir, dir_path)).mkdir(parents=True, exist_ok=True)

    classes_list = []
    class_to_id = {}
    processed_count = 0
    skipped_count = 0
    dir_counts = defaultdict(int)
    vae_dirs = [f"VAE_1118_obj{i}" for i in range(2, 9)]

    print("Collecting unique classes...")
    for vae_dir in os.listdir(base_dir):
        if vae_dir not in vae_dirs:
            continue

        labels_dir = os.path.join(base_dir, vae_dir, 'labels_after_0')
        if not os.path.isdir(labels_dir):
            continue

        for name_file in os.listdir(labels_dir):
            if not name_file.startswith('num_') or not name_file.endswith('_name.txt'):
                continue

            with open(os.path.join(labels_dir, name_file), 'r') as f:
                for line in f:
                    for class_name in line.strip().split():
                        if class_name not in class_to_id:
                            class_to_id[class_name] = len(classes_list)
                            classes_list.append(class_name)

    print(f"Found {len(classes_list)} unique classes")

    for vae_dir in vae_dirs:
        if not os.path.exists(os.path.join(base_dir, vae_dir)):
            continue

        print(f"\nProcessing directory: {vae_dir}")
        num_objects = int(vae_dir.split('obj')[-1])

        images_dir = os.path.join(base_dir, vae_dir, 'origin_images_after')
        labels_dir = os.path.join(base_dir, vae_dir, 'labels_after_0')

        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            continue

        for img_file in sorted(os.listdir(images_dir)):
            if not img_file.endswith('_0.png'):
                continue

            try:
                img_id = int(img_file.split('_')[1])
                batch_num = ((img_id // 20)) * 20 + 20
                if batch_num > max_batch:
                    skipped_count += 1
                    continue

                name_pattern = f"num_{num_objects}_{batch_num}_name.txt"
                pos_pattern = f"num_{num_objects}_{batch_num}.txt"
                name_file = os.path.join(labels_dir, name_pattern)
                pos_file = os.path.join(labels_dir, pos_pattern)

                if not os.path.exists(name_file) or not os.path.exists(pos_file):
                    continue

                with open(name_file, 'r') as f:
                    class_lines = f.readlines()
                with open(pos_file, 'r') as f:
                    pos_lines = f.readlines()

                line_idx = img_id % 20
                if line_idx >= len(class_lines) or line_idx >= len(pos_lines):
                    continue

                class_names = class_lines[line_idx].strip().split()
                pos_values = [float(x) for x in pos_lines[line_idx].strip().split()]

                img_path = os.path.join(images_dir, img_file)
                detected_boxes = detector.detect_objects(img_path)

                new_annotations = match_objects_to_labels(
                    detected_boxes,
                    class_names,
                    pos_values,
                    class_to_id
                )

                if new_annotations:
                    is_train = random.random() < train_ratio
                    subset = 'train' if is_train else 'val'

                    dst_img = os.path.join(output_dir, 'images', subset, f"{vae_dir}_{img_file}")
                    shutil.copy2(img_path, dst_img)
                    label_path = os.path.join(output_dir, 'labels', subset,
                                            f"{vae_dir}_{img_file.replace('.png', '.txt')}")
                    with open(label_path, 'w') as f:
                        for ann in new_annotations:
                            line = f"{int(ann[0])} {' '.join(f'{x:.6f}' for x in ann[1:])}\n"
                            f.write(line)

                    processed_count += 1
                    dir_counts[vae_dir] += 1

                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} images...")
                else:
                    skipped_count += 1

            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue

    print("\nProcessing Summary:")
    print(f"Total processed: {processed_count}")
    print(f"Total skipped: {skipped_count}")
    for dir_name in sorted(dir_counts.keys()):
        print(f"{dir_name}: {dir_counts[dir_name]} images")

    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        f.write('\n'.join(classes_list))

    return classes_list

if __name__ == "__main__":
    classes = create_yolo_dataset(
        base_dir="../../../../../Desktop/data/dataset",
        output_dir="yolo_dataset"
    )
    print("\nClasses:", classes)
