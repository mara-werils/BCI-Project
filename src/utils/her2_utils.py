"""
Utility functions for HER2 label extraction and handling.
"""

import re
from pathlib import Path


# HER2 class mapping
HER2_CLASSES = ['0', '1+', '2+', '3+']
HER2_TO_IDX = {'0': 0, '1+': 1, '2+': 2, '3+': 3}
IDX_TO_HER2 = {0: '0', 1: '1+', 2: '2+', 3: '3+'}


def extract_her2_from_filename(filename):
    """
    Extract HER2 label from filename.
    
    Args:
        filename: Image filename (e.g., '00000_train_1+.png')
    
    Returns:
        str: HER2 level ('0', '1+', '2+', '3+') or None if not found
    """
    if isinstance(filename, Path):
        stem = filename.stem
    else:
        stem = Path(filename).stem
    
    # Match pattern: ends with _0, _1+, _2+, or _3+
    match = re.search(r'_([0-3]\+?)$', stem)
    if match:
        return match.group(1)
    return None


def her2_to_index(her2_label):
    """
    Convert HER2 label string to class index.
    
    Args:
        her2_label: HER2 level string ('0', '1+', '2+', '3+')
    
    Returns:
        int: Class index (0-3) or -1 if invalid
    """
    return HER2_TO_IDX.get(her2_label, -1)


def index_to_her2(idx):
    """
    Convert class index to HER2 label string.
    
    Args:
        idx: Class index (0-3)
    
    Returns:
        str: HER2 level string or None if invalid
    """
    return IDX_TO_HER2.get(idx, None)


def get_her2_label_from_path(image_path):
    """
    Get HER2 class index from image path.
    
    Args:
        image_path: Full path or filename of image
    
    Returns:
        int: Class index (0-3) or -1 if not found
    """
    her2_str = extract_her2_from_filename(image_path)
    if her2_str is None:
        return -1
    return her2_to_index(her2_str)


def get_class_weights(labels, num_classes=4):
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        labels: List of class indices
        num_classes: Number of classes
    
    Returns:
        list: Weight for each class (inverse frequency)
    """
    import numpy as np
    
    counts = np.zeros(num_classes)
    for label in labels:
        if 0 <= label < num_classes:
            counts[label] += 1
    
    # Inverse frequency weighting
    total = sum(counts)
    weights = [total / (num_classes * c) if c > 0 else 0.0 for c in counts]
    
    return weights


def analyze_dataset_distribution(image_dir):
    """
    Analyze HER2 distribution in dataset.
    
    Args:
        image_dir: Path to image directory
    
    Returns:
        dict: Statistics about dataset
    """
    from collections import Counter
    
    image_dir = Path(image_dir)
    labels = []
    
    for img_path in image_dir.glob('*.png'):
        label = extract_her2_from_filename(img_path.name)
        if label:
            labels.append(label)
    
    counter = Counter(labels)
    total = len(labels)
    
    stats = {
        'total': total,
        'distribution': dict(counter),
        'percentages': {k: v/total*100 for k, v in counter.items()} if total > 0 else {}
    }
    
    return stats


if __name__ == '__main__':
    # Test functions
    print("Testing HER2 utils...")
    
    # Test extraction
    test_files = [
        '00000_train_1+.png',
        '00015_train_0.png', 
        '00006_train_2+.png',
        '00001_train_3+.png'
    ]
    
    for f in test_files:
        her2 = extract_her2_from_filename(f)
        idx = her2_to_index(her2)
        print(f"{f} -> HER2: {her2}, Index: {idx}")
    
    print("\nHER2 classes:", HER2_CLASSES)
    print("Mapping:", HER2_TO_IDX)


