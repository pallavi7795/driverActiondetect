import os
import sys
import shutil
import argparse
import logging
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define SageMaker Processing paths
INPUT_IMAGES_PATH = "/opt/ml/processing/input/images"
INPUT_LABELS_PATH = "/opt/ml/processing/input/labels"
CONFIG_PATH = "/opt/ml/processing/input/config"
OUTPUT_PATH = "/opt/ml/processing/output"

def verify_input_data():
    """Verify input data exists and has correct structure"""
    try:
        # Check if directories exist
        if not os.path.exists(INPUT_IMAGES_PATH):
            raise FileNotFoundError(f"Images directory not found at: {INPUT_IMAGES_PATH}")
        if not os.path.exists(INPUT_LABELS_PATH):
            raise FileNotFoundError(f"Labels directory not found at: {INPUT_LABELS_PATH}")

        dataset_info = {'images': {}, 'labels': {}}
        splits = ['train', 'validation', 'test']

        # Verify each split
        for split in splits:
            # Check split directories
            img_split_path = os.path.join(INPUT_IMAGES_PATH, split)
            label_split_path = os.path.join(INPUT_LABELS_PATH, split)

            if not os.path.exists(img_split_path):
                raise FileNotFoundError(f"Images {split} directory not found at: {img_split_path}")
            if not os.path.exists(label_split_path):
                raise FileNotFoundError(f"Labels {split} directory not found at: {label_split_path}")

            # Count and verify files
            img_files = [f for f in os.listdir(img_split_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            label_files = [f for f in os.listdir(label_split_path) 
                          if f.endswith('.txt')]

            dataset_info['images'][split] = len(img_files)
            dataset_info['labels'][split] = len(label_files)

            logger.info(f"Found {len(img_files)} images and {len(label_files)} labels in {split}")

            # Verify matching files
            img_basenames = {os.path.splitext(f)[0] for f in img_files}
            label_basenames = {os.path.splitext(f)[0] for f in label_files}

            missing_labels = img_basenames - label_basenames
            missing_images = label_basenames - img_basenames

            if missing_labels:
                logger.warning(f"Missing labels for images in {split}: {missing_labels}")
            if missing_images:
                logger.warning(f"Missing images for labels in {split}: {missing_images}")

        return dataset_info

    except Exception as e:
        logger.error(f"Error verifying input data: {str(e)}")
        raise

def load_config():
    """Load configuration file"""
    try:
        config_file = os.path.join(CONFIG_PATH, 'config.yaml')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Loaded configuration file")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def create_data_yaml(config):
    """Create YOLO data configuration file"""
    try:
        data_yaml = {
            'path': OUTPUT_PATH,
            'train': 'images/train',
            'val': 'images/validation',
            'test': 'images/test',
            'nc': config['nc'],
            'names': config['names']
        }

        data_yaml_path = os.path.join(OUTPUT_PATH, 'data.yaml')
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, sort_keys=False)

        logger.info(f"Created data.yaml at: {data_yaml_path}")
    except Exception as e:
        logger.error(f"Error creating data.yaml: {str(e)}")
        raise

def process_split(split, input_images, input_labels, output_images, output_labels):
    """Process a specific data split"""
    try:
        # Create split directories
        os.makedirs(os.path.join(output_images, split), exist_ok=True)
        os.makedirs(os.path.join(output_labels, split), exist_ok=True)

        # Get files
        img_files = [f for f in os.listdir(os.path.join(input_images, split)) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Process each image and its corresponding label
        for img_file in img_files:
            base_name = os.path.splitext(img_file)[0]
            label_file = f"{base_name}.txt"

            # Copy image
            src_img = os.path.join(input_images, split, img_file)
            dst_img = os.path.join(output_images, split, img_file)
            shutil.copy2(src_img, dst_img)

            # Copy label if exists
            src_label = os.path.join(input_labels, split, label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(output_labels, split, label_file)
                shutil.copy2(src_label, dst_label)

        logger.info(f"Processed {len(img_files)} files for {split} split")

    except Exception as e:
        logger.error(f"Error processing {split} split: {str(e)}")
        raise

def process_data():
    """Process the input data"""
    try:
        # Create output directory structure
        output_images = os.path.join(OUTPUT_PATH, 'images')
        output_labels = os.path.join(OUTPUT_PATH, 'labels')
        os.makedirs(output_images, exist_ok=True)
        os.makedirs(output_labels, exist_ok=True)

        # Process each split
        splits = ['train', 'validation', 'test']
        for split in splits:
            process_split(split, INPUT_IMAGES_PATH, INPUT_LABELS_PATH, 
                        output_images, output_labels)

        logger.info("Data processing completed")

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

def main():
    try:
        # Verify input data
        logger.info("Verifying input data...")
        dataset_info = verify_input_data()

        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()

        # Process data
        logger.info("Processing data...")
        process_data()

        # Create data.yaml
        logger.info("Creating data.yaml...")
        create_data_yaml(config)

        logger.info("Preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
