import os
import sys
import yaml
from ultralytics import YOLO

# Ensure src module access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger

def validate_dataset_exists(images_path):
    """
    Validate that the dataset directory contains data.
    """
    if not os.path.exists(images_path):
        return False
        
    # Check if there are any files in the train folder
    files = os.listdir(images_path)
    return len(files) > 0

def load_config(config_path):
    """Load YAML configuration."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        sys.exit(1)

def main():
    config_path = "configs/config.yaml"
    logger.info("Initializing Custom YOLOv8 Training Pipeline...")
    
    config = load_config(config_path)
    train_cfg = config.get("training", {})
    
    if not train_cfg:
        logger.error("No 'training' block found in config.yaml.")
        sys.exit(1)
        
    data_yaml = train_cfg.get("data_yaml", "configs/dataset.yaml")
    epochs = train_cfg.get("epochs", 50)
    batch_size = train_cfg.get("batch_size", 16)
    img_size = train_cfg.get("img_size", 640)
    project_dir = train_cfg.get("project_dir", "models/yolov8")
    name = train_cfg.get("name", "custom_model")
    
    # Standard base model to initialize weights
    base_model = "yolov8n.pt"
    
    # 1. Validation Phase
    train_images_path = os.path.join("data", "dataset", "images", "train")
    if not validate_dataset_exists(train_images_path):
         logger.error(f"No training images found in '{train_images_path}'!")
         logger.error("Please drop your annotated image dataset into the training folders before launching.")
         sys.exit(1)
         
    # 2. Training Phase
    logger.info(f"Loading Base Model ({base_model})...")
    model = YOLO(base_model)
    
    logger.info(f"Starting training run '{name}' for {epochs} epochs...")
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=project_dir,
            name=name,
            exist_ok=True # Allows resume/overwrite
        )
        logger.info(f"Training completed successfully! Saved to {project_dir}/{name}/weights/best.pt")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
