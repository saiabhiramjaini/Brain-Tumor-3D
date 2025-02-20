import nibabel as nib
import numpy as np
from mayavi import mlab

# Load NIfTI file
def load_nifti(file_path):
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata()

# 3D Visualization with Color Grading and Transparency
def visualize_3d_mri(mri_data, brain_threshold=0.1, tumor_threshold=0.6, brain_opacity=0.1):
    mri_data = mri_data / np.max(mri_data)  # Normalize data
    
    # Create figure
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1000, 800))
    
    # Visualize brain tissue with low opacity (transparent gray)
    brain_mask = (mri_data > brain_threshold) & (mri_data <= tumor_threshold)
    x_brain, y_brain, z_brain = np.where(brain_mask)
    
    # For brain, use a transparent gray colormap
    brain_pts = mlab.points3d(
        x_brain, y_brain, z_brain, 
        mri_data[x_brain, y_brain, z_brain],
        mode="cube", 
        colormap="bone",  # Gray colormap
        scale_factor=1,
        opacity=brain_opacity,  # Set low opacity for brain tissue
        vmin=brain_threshold,
        vmax=tumor_threshold
    )
    
    # Visualize tumor with high opacity and different color
    tumor_mask = mri_data > tumor_threshold
    x_tumor, y_tumor, z_tumor = np.where(tumor_mask)
    
    if len(x_tumor) > 0:  # Check if tumor exists
        tumor_pts = mlab.points3d(
            x_tumor, y_tumor, z_tumor,
            mri_data[x_tumor, y_tumor, z_tumor],
            mode="cube",
            colormap="hot",  # Red-yellow colormap for tumor
            scale_factor=1,
            opacity=1.0,  # Full opacity for tumor
            vmin=tumor_threshold,
            vmax=1.0
        )
        
    #     # Add tumor colorbar
    #     tumor_cb = mlab.colorbar(tumor_pts, title="Tumor Intensity", orientation="vertical", nb_labels=5)
    
    # # Add brain colorbar
    # brain_cb = mlab.colorbar(brain_pts, title="Brain Tissue", orientation="vertical", nb_labels=5)
    
    
    # Show the visualization
    mlab.show()

# Example usage
file_path = "BraTS19_2013_0_1_t1.nii"  # Change to your actual file
mri_data = load_nifti(file_path)

# Visualize with adjustable parameters
visualize_3d_mri(
    mri_data,
    brain_threshold=0.1,   # Adjust this to capture enough brain tissue
    tumor_threshold=0.6,   # Adjust this based on your tumor intensity
    brain_opacity=0.1      # Adjust for desired brain transparency
)