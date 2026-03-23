"""
Dataset Balancing Script

Randomly samples a target number of images from each class to create a balanced dataset.
"""

import os
import random
import shutil
import argparse
from pathlib import Path


def balance_dataset(
    source_path: str,
    balance_dir_name: str = "balance_data",
    target_count: int = 7200,
    seed: int = 42,
    copy: bool = True
):
    """
    Balance dataset by randomly sampling target_count images from each class.
    
    Args:
        source_path: Path to raw dataset with class subdirectories
        balance_dir_name: Name of the output balanced dataset directory
        target_count: Number of images to sample per class
        seed: Random seed for reproducibility
        copy: If True, copy images; if False, create symlinks
    """
    random.seed(seed)
    source_path = Path(source_path)
    balance_path = source_path / balance_dir_name
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    print(f"Source: {source_path}")
    print(f"Target: {balance_path}")
    print(f"Target count per class: {target_count}")
    print(f"Random seed: {seed}")
    
    # Remove existing balanced dataset
    if balance_path.exists():
        print(f"\nRemoving existing {balance_path}")
        shutil.rmtree(balance_path)
    
    balance_path.mkdir(parents=True, exist_ok=True)
    
    # Process each class directory
    results = {}
    class_dirs = [d for d in source_path.iterdir() if d.is_dir() and d.name != balance_dir_name]
    
    for class_source in sorted(class_dirs):
        class_name = class_source.name
        class_target = balance_path / class_name
        
        # Get all image files
        image_files = [
            f for f in class_source.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        available = len(image_files)
        print(f"\n📁 {class_name}:")
        print(f"   Available: {available:,} images")
        
        # Handle insufficient images
        if available < target_count:
            print(f"   ⚠️  Not enough images! Need {target_count}, have {available}")
            selected = image_files
        else:
            selected = random.sample(image_files, target_count)
            print(f"   ✓ Selected: {len(selected):,} images")
        
        results[class_name] = len(selected)
        
        # Create target directory
        class_target.mkdir(parents=True, exist_ok=True)
        
        # Transfer selected images
        transferred = 0
        for img_path in selected:
            target_file = class_target / img_path.name
            
            if copy:
                shutil.copy2(img_path, target_file)
            else:
                target_file.symlink_to(img_path.resolve())
            
            transferred += 1
            
            # Progress indicator
            if transferred % 1000 == 0:
                print(f"   ... {transferred:,} images processed")
        
        action = "Copied" if copy else "Linked"
        print(f"   ✓ {action}: {transferred:,} images")
    
    # Verification
    print("\n" + "=" * 50)
    print("📊 VERIFICATION - Balanced Dataset Counts:")
    print("=" * 40)
    
    total = 0
    for class_dir in sorted(balance_path.iterdir()):
        if class_dir.is_dir():
            count = len([f for f in class_dir.iterdir() if f.is_file()])
            print(f"  {class_dir.name}: {count:,} images")
            total += count
    
    print("-" * 40)
    print(f"  TOTAL: {total:,} images")
    print("=" * 40)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Balance dataset by sampling equal images per class"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/home/moeenuddin/Desktop/Deep_learning/ocean/Ocean-_Clean_vs_Ocean_Trash_Classification/dataset/raw",
        help="Path to raw dataset directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="balance_data",
        help="Name of output balanced dataset directory"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=7200,
        help="Target number of images per class"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying files"
    )
    
    args = parser.parse_args()
    
    balance_dataset(
        source_path=args.source,
        balance_dir_name=args.output,
        target_count=args.count,
        seed=args.seed,
        copy=not args.symlink
    )


if __name__ == "__main__":
    main()
