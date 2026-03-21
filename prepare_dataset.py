import os
import pandas as pd
from PIL import Image

# Dictionary identifying which type a specific model belongs to.
# 0: Civil Passenger, 1: Military Fighter, 2: Military Transport/Bomber
MODEL_TO_TYPE_MAPPING = {
    'Boeing_737': 0,
    'Airbus_A320': 0,
    'Boeing_777': 0,
    'Su-27': 1,
    'F-16': 1,
    'MiG-29': 1,
    'C-130_Hercules': 2,
    'Il-76': 2
}

def validate_and_prepare_dataset(base_dir='dataset_images'):
    """
    Scans the downloaded dataset folders, validates images using Pillow, 
    and generates a multi-label CSV mapping (File, Model, Type).
    This fulfills the requirement of data preparation using Pandas and Pillow.
    """
    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' not found. Creating placeholder directories...")
        os.makedirs(base_dir)
        # Create dummy folders for the user to drop images into later
        for model in MODEL_TO_TYPE_MAPPING.keys():
            os.makedirs(os.path.join(base_dir, model), exist_ok=True)
        print("Done! Please place the downloaded Kaggle images into the respective model folders inside 'dataset_images'.")
        print("Then run this script again to generate the metadata CSV.")
        return

    data_records = []
    
    # Iterate over model folders
    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue
            
        # Get the corresponding aircraft type from mapping
        aircraft_type = MODEL_TO_TYPE_MAPPING.get(model_name, -1)
        if aircraft_type == -1:
            print(f"Warning: Model {model_name} is not in the TYPE mapping dictionary. Skipping.")
            continue
            
        print(f"Processing folder: {model_name} (Type ID: {aircraft_type})")
        
        # Iterate over images in the model folder
        for filename in os.listdir(model_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            file_path = os.path.join(model_path, filename)
            
            # Validate image integrity with Pillow
            try:
                with Image.open(file_path) as img:
                    img.verify() # Verify it's a valid and uncorrupted image
                
                # Add to records for Pandas
                data_records.append({
                    'image_path': file_path,
                    'model_label': model_name,
                    'type_label': aircraft_type
                })
            except Exception as e:
                print(f"Bad or corrupted image skipped: {file_path} - {e}")
                
    if not data_records:
        print("\nNo valid images found to process. Please download them and place them in the folders.")
        return
        
    # Create a Pandas DataFrame
    df = pd.DataFrame(data_records)
    
    # Save the prepared metadata to CSV
    csv_path = 'dataset_metadata.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSuccessfully prepared dataset metadata for {len(df)} images.")
    print(f"Saved to: {csv_path}")

if __name__ == "__main__":
    validate_and_prepare_dataset()
