import streamlit as st
import nibabel as nib
import numpy as np
from mayavi import mlab
import tempfile
import os
from tvtk.api import tvtk
from pyface.api import GUI

# Configure Mayavi to use the Qt backend
import vtk
from mayavi.core.api import Engine
from mayavi.core.off_screen_engine import OffScreenEngine

def load_nifti(file_path):
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata()

def save_visualization(mri_data, brain_threshold=0.1, tumor_threshold=0.6, brain_opacity=0.1):
    # Create an off-screen engine
    engine = OffScreenEngine()
    engine.start()
    
    # Create a new scene
    scene = engine.new_scene()
    scene.scene.off_screen_rendering = True
    
    # Normalize data
    mri_data = mri_data / np.max(mri_data)
    
    # Create the visualization
    brain_mask = (mri_data > brain_threshold) & (mri_data <= tumor_threshold)
    x_brain, y_brain, z_brain = np.where(brain_mask)
    
    brain_pts = mlab.points3d(
        x_brain, y_brain, z_brain, 
        mri_data[x_brain, y_brain, z_brain],
        mode="cube", colormap="bone", scale_factor=1,
        opacity=brain_opacity, vmin=brain_threshold, vmax=tumor_threshold,
        figure=scene.mayavi_scene
    )
    
    tumor_mask = mri_data > tumor_threshold
    x_tumor, y_tumor, z_tumor = np.where(tumor_mask)
    
    if len(x_tumor) > 0:
        tumor_pts = mlab.points3d(
            x_tumor, y_tumor, z_tumor,
            mri_data[x_tumor, y_tumor, z_tumor],
            mode="cube", colormap="hot", scale_factor=1,
            opacity=1.0, vmin=tumor_threshold, vmax=1.0,
            figure=scene.mayavi_scene
        )
    
    # Save the visualization
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, 'visualization.png')
    scene.scene.save(output_path)
    
    # Clean up
    engine.stop()
    return output_path

st.set_page_config(page_title="3D Brain MRI Visualizer", layout="wide")
st.title("🧠 3D Brain MRI Visualizer")

# Add description
st.markdown("""
This app allows you to visualize 3D brain MRI data. Upload a NIFTI (.nii) file and adjust the visualization parameters.
- **Brain Threshold**: Controls the minimum intensity value for brain tissue visualization
- **Tumor Threshold**: Controls the threshold for potential tumor visualization
- **Brain Opacity**: Adjusts the transparency of brain tissue
""")

uploaded_file = st.file_uploader("Upload a .nii file", type=["nii"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    
    st.success("File uploaded successfully!")
    
    try:
        mri_data = load_nifti(temp_path)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            brain_threshold = st.slider("Brain Threshold", 0.01, 1.0, 0.1, 0.01)
        with col2:
            tumor_threshold = st.slider("Tumor Threshold", 0.1, 1.0, 0.6, 0.01)
        with col3:
            brain_opacity = st.slider("Brain Opacity", 0.01, 1.0, 0.1, 0.01)
        
        if st.button("Generate Visualization"):
            with st.spinner("Generating 3D visualization..."):
                try:
                    output_path = save_visualization(
                        mri_data, 
                        brain_threshold, 
                        tumor_threshold, 
                        brain_opacity
                    )
                    st.image(output_path, caption="3D Brain MRI Visualization")
                    os.remove(output_path)
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
        
    except Exception as e:
        st.error(f"Error loading NIFTI file: {str(e)}")
    
    os.remove(temp_path)

st.sidebar.markdown("""
### Instructions
1. Upload a NIFTI (.nii) file
2. Adjust the visualization parameters
3. Click 'Generate Visualization' to view the results

### About
This app uses Mayavi for 3D visualization of brain MRI data. It can help identify and visualize brain structures and potential abnormalities.
""")