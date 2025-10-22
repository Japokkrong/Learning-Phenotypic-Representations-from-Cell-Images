import pandas as pd
import os

# === Configuration ===
input_csv = "BBBC021_DMSO.csv"  
output_csv =  "BBBC021_DMSO_fixed.csv"  

# input_csv = "BBBC021_annotated.csv"
# output_csv = "BBBC021_annotated_fixed.csv"

# The base directory where your real dataset is located
new_base = "../datasets/bbbc021"

# === Load CSV ===
df = pd.read_csv(input_csv)

# === Function to fix each path ===
def fix_path(path):
    if not isinstance(path, str):
        return path
    
    # Replace the old prefix
    if "/projects/img/GAN_CP/PAPER_2/BBBC021/" in path:
        
        # /projects/img/GAN_CP/PAPER_2/BBBC021/Week1_22123/Week1_150607_B04_s1_w11323931B-BDA7-4F42-870E-7174BFBF3643_corrected_resized.tiff
        # warissara/datasets/bbbc021/BBBC021_v1_images_Week1_22123/Week1_22123/Week1_150607_B02_s1_w107447158-AC76-4844-8431-E6A954BD1174.tif
        # warissara/datasets/bbbc021/BBBC021_v1_images_Week1_22141/Week1_22141/Week1_150607_E02_s1_w17B32AA6E-8874-448B-A56E-9F032278638F.tif
        
        # warissara/datasets/bbbc021/BBBC021_v1_images_Week1_22141/Week1_22141/Week1_150607_E02_s1_w17B32AA6E-8874-448B-A56E-9F032278638F_corrected_resized.tif
        
        # '/home/ravipas.aph/warissara/datasets/bbbc021/Week1_22141/Week1_150607_E02_s1_w17B32AA6E-8874-448B-A56E-9F032278638F_corrected_resized.tif'
        
        # Keep only the part after "BBBC021/"
        relative = path.split("BBBC021/")[-1]
        bbbc_path = "BBBC021_v1_images_" + relative.split("/")[0]
        new_path = os.path.join(new_base, bbbc_path, relative)
    else:
        new_path = path
    
    # Replace .tiff -> .tif
    new_path = new_path.replace(".tiff", ".tif")
    new_path = new_path.replace("_corrected_resized", "")

    return new_path

# === Apply to all columns that look like paths ===
for col in df.columns:
    if "Path" in col or "FileName" in col:
        df[col] = df[col].apply(fix_path)

# === Save result ===
df.to_csv(output_csv, index=False)
print(f"✅ Fixed paths saved to: {output_csv}")
