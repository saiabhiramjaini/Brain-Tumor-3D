import streamlit as st
import nibabel as nib
import numpy as np
import plotly.graph_objs as go
import tempfile
import os

def load_nifti(file_path):
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata()

def normalize_data(data):
    """Normalize data to 0-1 range with better handling of outliers"""
    p1, p99 = np.percentile(data, (1, 99))
    data_clip = np.clip(data, p1, p99)
    return (data_clip - p1) / (p99 - p1)

def visualize_3d_mri(mri_data):
    # Normalize data with better outlier handling
    mri_data = normalize_data(mri_data)
    
    # Fixed thresholds for brain and tumor
    brain_threshold = 0.2
    tumor_threshold = 0.6
    
    # Create masks for brain and tumor
    brain_mask = (mri_data > brain_threshold) & (mri_data <= tumor_threshold)
    tumor_mask = mri_data > tumor_threshold
    
    # Get coordinates for brain and tumor
    x_brain, y_brain, z_brain = np.where(brain_mask)
    x_tumor, y_tumor, z_tumor = np.where(tumor_mask)
    
    scale = 1.0 
    
    # Create 3D scatter plots
    brain_scatter = go.Scatter3d(
        x=x_brain * scale,
        y=y_brain * scale,
        z=z_brain * scale,
        mode="markers",
        marker=dict(
            size=4, 
            color="lightblue",
            opacity=0.3,
            line=dict(width=0)
        ),
        name="Brain Tissue"
    )
    
    tumor_scatter = go.Scatter3d(
        x=x_tumor * scale,
        y=y_tumor * scale,
        z=z_tumor * scale,
        mode="markers",
        marker=dict(
            size=0.1,  
            color="rgba(255, 255, 255, 0.5)", 
            opacity=0, 
            line=dict(width=0)
        ),
        name=" "
    )
    
    # Create layout with better camera angle and larger initial view
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5) 
            ),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
        legend=dict(
            x=0.7,
            y=0.9,
            bgcolor="rgba(255, 255, 255, 0.5)"
        )
    )
    
    fig = go.Figure(data=[brain_scatter, tumor_scatter], layout=layout)
    
    # Update the scene range to ensure proper initial scaling
    max_range = max(
        max(x_brain * scale) - min(x_brain * scale),
        max(y_brain * scale) - min(y_brain * scale),
        max(z_brain * scale) - min(z_brain * scale)
    )
    
    fig.update_layout(
        scene=dict(
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='manual',
            xaxis=dict(range=[-max_range/2, max_range/2]),
            yaxis=dict(range=[-max_range/2, max_range/2]),
            zaxis=dict(range=[-max_range/2, max_range/2])
        )
    )
    
    return fig

def main():
    st.title("3D Brain Visualization")
    
    uploaded_file = st.file_uploader("Upload a NIfTI file", type=["nii"])
    
    if uploaded_file is not None:
        with st.spinner("Loading NIfTI file..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
            
            mri_data = load_nifti(file_path)
            
            if st.button("Visualize"):
                with st.spinner("Generating 3D visualization..."):
                    fig = visualize_3d_mri(mri_data)
                    st.plotly_chart(fig, use_container_width=True)
            
            os.unlink(file_path)

if __name__ == "__main__":
    main()