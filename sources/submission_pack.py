import os
import numpy as np
import pandas as pd

def create_submission_pack(processed_dir='data/processed', sample_dir='submission_sample'):
    """
    Creates a representative sample pack of exactly 100 windows/rows per dataset
    as required by Section 8 of the assessment brief.
    """
    print(f"\n--- Generating 100-Sample Submission Pack ---")
    
    # Ensure the target directory exists
    os.makedirs(sample_dir, exist_ok=True)
    
    if not os.path.exists(processed_dir):
        print(f"[FAIL] Processed directory not found: {processed_dir}")
        return

    # Grab all processed files
    files = [f for f in sorted(os.listdir(processed_dir)) if f.endswith(('.npz', '.csv'))]
    
    for file_name in files:
        # Skip the manifest itself if it's in there
        if 'manifest' in file_name.lower():
            continue
            
        in_path = os.path.join(processed_dir, file_name)
        out_path = os.path.join(sample_dir, file_name)
        
        # 1. Handle NumPy Arrays (Signals)
        if file_name.endswith('.npz'):
            data = np.load(in_path)
            if 'X' in data:
                # Slice exactly the first 100 samples along the 0th axis
                X_sample = data['X'][:100]
                
                # Check if there are labels (y) attached to this file
                if 'y' in data:
                    y_sample = data['y'][:100]
                    np.savez_compressed(out_path, X=X_sample, y=y_sample)
                else:
                    np.savez_compressed(out_path, X=X_sample)
                
                print(f" -> Saved {file_name}: Signal shape {X_sample.shape}")
                
        # 2. Handle CSV Files (Metadata)
        elif file_name.endswith('.csv'):
            df = pd.read_csv(in_path)
            # Slice exactly the first 100 rows
            df_sample = df.head(100)
            df_sample.to_csv(out_path, index=False)
            
            print(f" -> Saved {file_name}: Metadata shape {df_sample.shape}")

    print("\n[SUCCESS] Sample pack generated in 'submission_sample/'")

# Run the function
if __name__ == "__main__":
    create_submission_pack()
