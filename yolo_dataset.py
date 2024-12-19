import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

def get_num_objects_from_dir(dir_name):
    """Extract number of objects from directory name (e.g., 'VAE_1118_obj5' -> 5)"""
    return int(dir_name.split('obj')[-1])

def find_label_batch(labels_dir, img_id, num_objects, max_batch=2200):
    """Find the correct batch files for an image ID."""
    if img_id >= max_batch:
        return None

    batch_num = ((img_id // 20)) * 20 + 20
    if batch_num > max_batch:
        return None

    name_pattern = f"num_{num_objects}_{batch_num}_name.txt"
    pos_pattern = f"num_{num_objects}_{batch_num}.txt"

    name_file = os.path.join(labels_dir, name_pattern)
    pos_file = os.path.join(labels_dir, pos_pattern)

    if os.path.exists(name_file) and os.path.exists(pos_file):
        return batch_num
    return None

def analyze_additional_values(pos_values):
    """Analyze the patterns in the additional values for each object"""
    num_objects = len(pos_values) // 7
    print("\nAnalyzing additional values:")
    
    for i in range(num_objects):
        base_idx = i * 7
        x = float(pos_values[base_idx])
        y = float(pos_values[base_idx + 1])
        w = float(pos_values[base_idx + 2])
        h = float(pos_values[base_idx + 3])
        v5 = float(pos_values[base_idx + 4])
        v6 = float(pos_values[base_idx + 5])
        v7 = float(pos_values[base_idx + 6])
        
        print(f"Object {i+1}:")
        print(f"  Main coords: x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f}")
        print(f"  Extra values: v5={v5:.4f}, v6={v6:.0f}, v7={v7:.0f}")

def process_coordinates(pos_values, pos_idx, debug=False):
    """
    Simple YOLO conversion maintaining exact proportions
    Input: [y, x, h, w] in 0.27/0.28 space
    Output: [x_center, y_center, w, h] in 0-1 space
    """
    try:
        # Get values (y, x, h, w)
        y = float(pos_values[pos_idx])
        x = float(pos_values[pos_idx + 1])
        h = float(pos_values[pos_idx + 2])
        w = float(pos_values[pos_idx + 3])

        v5 = float(pos_values[pos_idx + 4])
        v6 = float(pos_values[pos_idx + 5])
        v7 = float(pos_values[pos_idx + 6])

        # Constants
        # MAX_X = 0.27
        # MAX_Y = 0.28
        MAX_X = 0.4
        MAX_Y = 0.35


        # 1. First normalize all values to 0-1 space
        my = -MAX_Y if v7%2==0 else 0
        mx = MAX_X if v7%2==0 else 0
        x_norm = (x) / MAX_X
        y_norm = (y) / MAX_Y
        w_norm = (w) / MAX_X
        h_norm = (h) / MAX_Y

        # 2. Convert to center format AFTER normalization
        x_center = x_norm
        y_center = y_norm

        if debug:
            print(f"\nObject Analysis:")
            print(f"Raw input: y={y:.6f}, x={x:.6f}, h={h:.6f}, w={w:.6f}")
            print(f"Normalized box: x={x_norm:.6f}, y={y_norm:.6f}, w={w_norm:.6f}, h={h_norm:.6f}")
            print(f"Final center: x={x_center:.6f}, y={y_center:.6f}")

        return x_center, y_center, w_norm, h_norm, v5, v6, v7

    except (ValueError, IndexError) as e:
        raise ValueError(f"Error processing coordinates at index {pos_idx}: {str(e)}")

def create_yolo_dataset(base_dir, output_dir, train_ratio=0.8, max_batch=100, min_obj=2, max_obj=8):
    """Create a YOLO dataset from the given directory structure."""
    # Create output directory structure
    yolo_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for dir_path in yolo_dirs:
        Path(os.path.join(output_dir, dir_path)).mkdir(parents=True, exist_ok=True)

    # Track unique class names (keeping the numbers)
    classes_list = []  # Use a list to maintain order
    class_to_id = {}   # Map class names to their IDs
    processed_count = 0
    skipped_count = 0
    dir_counts = defaultdict(int)

    vae_dirs = [f"VAE_1118_obj{i}" for i in range(min_obj, max_obj + 1)]

    # First pass: collect all unique class names
    print("Collecting unique classes...")
    for vae_dir in os.listdir(base_dir):
        if vae_dir not in vae_dirs:
            continue

        vae_path = os.path.join(base_dir, vae_dir)
        labels_dir = os.path.join(vae_path, 'labels_after_0')

        if not os.path.isdir(labels_dir):
            continue

        # Look through all name files in this directory
        for name_file in os.listdir(labels_dir):
            if not name_file.startswith('num_') or not name_file.endswith('_name.txt'):
                continue

            with open(os.path.join(labels_dir, name_file), 'r') as f:
                for line in f:
                    for class_name in line.strip().split():
                        if class_name not in class_to_id:
                            class_to_id[class_name] = len(classes_list)
                            classes_list.append(class_name)

    print(f"Found {len(classes_list)} unique classes: {classes_list}")

    # Process each VAE directory
    for vae_dir in os.listdir(base_dir):
        if vae_dir not in vae_dirs:
            continue

        print(f"\nProcessing directory: {vae_dir}")
        num_objects = get_num_objects_from_dir(vae_dir)

        vae_path = os.path.join(base_dir, vae_dir)
        images_dir = os.path.join(vae_path, 'origin_images_after')
        labels_dir = os.path.join(vae_path, 'labels_after_0')

        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            continue

        # Process each image
        for img_file in sorted(os.listdir(images_dir)):
            if not img_file.endswith('_0.png'):
                continue

            try:
                img_id = int(img_file.split('_')[1])

                if img_id >= max_batch:
                    skipped_count += 1
                    continue

                batch_num = find_label_batch(labels_dir, img_id, num_objects, max_batch)

                if batch_num is None:
                    continue

                # Construct label filenames
                name_pattern = f"num_{num_objects}_{batch_num}_name.txt"
                pos_pattern = f"num_{num_objects}_{batch_num}.txt"

                name_file = os.path.join(labels_dir, name_pattern)
                pos_file = os.path.join(labels_dir, pos_pattern)

                # Read class names and positions
                with open(name_file, 'r') as f:
                    class_lines = f.readlines()
                with open(pos_file, 'r') as f:
                    pos_lines = f.readlines()

                # Find the correct line for this image
                line_idx = img_id % 20
                if line_idx >= len(pos_lines) or line_idx >= len(class_lines):
                    continue

                class_names = class_lines[line_idx].strip().split()
                pos_values = [float(x) for x in pos_lines[line_idx].strip().split()]

                # Analyze the position values if it's the first few images
                if processed_count < 100:
                    analyze_additional_values(pos_values)

                # Create YOLO format label
                yolo_label = []
                for i in range(len(class_names)):
                    pos_idx = i * 7
                    if pos_idx + 6 >= len(pos_values):
                        break

                    try:
                        class_id = class_to_id[class_names[i]]
                        x_center, y_center, w, h, v5, v6, v7 = process_coordinates(pos_values, pos_idx)
                        # Store all values including extra information
                        yolo_label.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {v5:.6f} {v6:.0f} {v7:.0f}")
                    except (ValueError, IndexError) as e:
                        print(f"Error processing object {i} in {img_file}: {str(e)}")
                        continue

                # Save image and label
                is_train = random.random() < train_ratio
                subset = 'train' if is_train else 'val'

                src_img = os.path.join(images_dir, img_file)
                dst_img = os.path.join(output_dir, 'images', subset, f"{vae_dir}_{img_file}")
                shutil.copy2(src_img, dst_img)

                label_path = os.path.join(output_dir, 'labels', subset, f"{vae_dir}_{img_file.replace('.png', '.txt')}")
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_label))

                processed_count += 1
                dir_counts[vae_dir] += 1

                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} images...")

            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue

        print(f"Processed {dir_counts[vae_dir]} images from {vae_dir}")

    print("\nProcessing Summary:")
    for dir_name in sorted(dir_counts.keys()):
        print(f"{dir_name}: {dir_counts[dir_name]} images")

    # Save the classes list
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        f.write('\n'.join(classes_list))

    return classes_list

if __name__ == "__main__":
    classes = create_yolo_dataset(
        base_dir="../../../../../Desktop/data/dataset",
        output_dir="yolo_dataset"
    )
    print("\nClasses:", classes)
