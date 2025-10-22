import pandas as pd

# Load your metadata
meta = pd.read_csv('../datasets/bbbc021/BBBC021_v1_image.csv')
moa = pd.read_csv('../datasets/bbbc021/BBBC021_v1_moa.csv')

# Clean names
meta['Image_Metadata_Compound'] = meta['Image_Metadata_Compound'].str.strip('"')
moa['compound'] = moa['compound'].str.strip('"')

# Merge to add MoA
merged = pd.merge(
    meta,
    moa,
    how='left',
    left_on=['Image_Metadata_Compound', 'Image_Metadata_Concentration'],
    right_on=['compound', 'concentration']
)

# Add missing columns with placeholder values
merged['Unique_MoA'] = 0
merged['Unique_Treatments'] = 0
merged['Unique_Compounds'] = 0

# If you can extract plate number (like Week4_27481 → 4)
merged['Plate'] = merged['Image_PathName_DAPI'].str.extract(r'Week(\d+)').fillna(0).astype(int)

# Batch could follow plate number or just 0
merged['Batch'] = merged['Plate']

# Rename columns to match GitHub format
merged.rename(columns={
    'moa': 'MoA'
}, inplace=True)

# Keep only the desired columns in correct order
final_cols = [
    'TableNumber', 'ImageNumber',
    'Image_FileName_DAPI', 'Image_PathName_DAPI',
    'Image_FileName_Tubulin', 'Image_PathName_Tubulin',
    'Image_FileName_Actin', 'Image_PathName_Actin',
    'Image_Metadata_Plate_DAPI', 'Image_Metadata_Well_DAPI',
    'Replicate', 'Image_Metadata_Compound', 'Image_Metadata_Concentration',
    'MoA', 'Unique_MoA', 'Unique_Treatments', 'Unique_Compounds', 'Plate', 'Batch'
]

final = merged[final_cols]

# Save
final.to_csv('full_formatted_dataset.csv', index=False)
print("✅ Saved as full_formatted_dataset.csv")
