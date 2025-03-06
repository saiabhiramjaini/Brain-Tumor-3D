import os
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
import tempfile

# -------------------------- HELPER FUNCTIONS --------------------------

def load_nifti(file_path):
    """Load a NIfTI file and return its data as a NumPy array."""
    nifti_img = nib.load(file_path)
    return np.asanyarray(nifti_img.dataobj)

def normalize_data(mri_data):
    """Normalize MRI data to range [0,1] for visualization."""
    return (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))

def overlay_tumor(ax, slice_data, tumor_threshold):
    """Overlay tumor mask on the given axis."""
    tumor_mask = slice_data > tumor_threshold
    if np.any(tumor_mask):
        tumor_overlay = np.zeros_like(slice_data)
        tumor_overlay[tumor_mask] = slice_data[tumor_mask]
        ax.imshow(tumor_overlay, cmap='hot', alpha=0.7, origin='lower')

def visualize_2d_slices(mri_data, brain_threshold, tumor_threshold):
    """Visualize middle slices (Sagittal, Coronal, Axial) of MRI scan."""
    mri_data = normalize_data(mri_data)
    x_mid, y_mid, z_mid = np.array(mri_data.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    views = [
        ("Sagittal", mri_data[x_mid, :, :].T, "Y", "Z"),
        ("Coronal", mri_data[:, y_mid, :].T, "X", "Z"),
        ("Axial", mri_data[:, :, z_mid], "X", "Y"),
    ]

    for ax, (title, data, xlabel, ylabel) in zip(axes, views):
        ax.imshow(data, cmap='gray', origin='lower')
        overlay_tumor(ax, data, tumor_threshold)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    return fig

def visualize_3d_mri(mri_data, brain_threshold, tumor_threshold, sample_ratio, point_size, brain_opacity):
    """Create a 3D visualization of MRI data using Plotly."""
    mri_data = normalize_data(mri_data)

    brain_mask = (mri_data > brain_threshold) & (mri_data <= tumor_threshold)
    tumor_mask = mri_data > tumor_threshold

    x_brain, y_brain, z_brain = np.where(brain_mask)
    brain_values = mri_data[x_brain, y_brain, z_brain]

    if len(x_brain) > 0:
        num_points = max(100, int(len(x_brain) * sample_ratio))
        if num_points < len(x_brain):
            idx = np.random.choice(len(x_brain), num_points, replace=False)
            x_brain, y_brain, z_brain = x_brain[idx], y_brain[idx], z_brain[idx]
            brain_values = brain_values[idx]

    x_tumor, y_tumor, z_tumor = np.where(tumor_mask)
    tumor_values = mri_data[x_tumor, y_tumor, z_tumor]

    if len(x_tumor) > 0:
        num_points = max(100, int(len(x_tumor) * sample_ratio * 2))
        if num_points < len(x_tumor):
            idx = np.random.choice(len(x_tumor), num_points, replace=False)
            x_tumor, y_tumor, z_tumor = x_tumor[idx], y_tumor[idx], z_tumor[idx]
            tumor_values = tumor_values[idx]

    brain_scatter = go.Scatter3d(
        x=x_brain, y=y_brain, z=z_brain,
        mode="markers",
        marker=dict(size=point_size, color=brain_values, colorscale='Blues', opacity=brain_opacity),
        name="Brain Tissue"
    )

    data = [brain_scatter]
    if len(x_tumor) > 0:
        tumor_scatter = go.Scatter3d(
            x=x_tumor, y=y_tumor, z=z_tumor,
            mode="markers",
            marker=dict(size=point_size * 1.2, color=tumor_values, colorscale='Hot', opacity=1.0),
            name="Potential Tumor"
        )
        data.append(tumor_scatter)

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title="X", showbackground=False),
            yaxis=dict(title="Y", showbackground=False),
            zaxis=dict(title="Z", showbackground=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        template="plotly_dark"
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

# -------------------------- STREAMLIT APP --------------------------

st.title("🧠 MRI Brain Visualization")

uploaded_file = st.file_uploader("Upload a NIfTI file (.nii or .nii.gz)", type=["nii", "nii.gz"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        mri_data = load_nifti(tmp_file_path)
        st.success(f"File loaded successfully! Shape: {mri_data.shape}")

        # Hardcoded Patient Details
        patient_info = {
            "Patient Name": "John Doe",
            "Age": "45 years",
            "Gender": "Male",
            "Patient ID": "JD20250306",
            "Scan Date": "2025-03-06",
            "Diagnosis": "Suspected Glioblastoma"
        }

        # Display Patient Details
        st.sidebar.header("🩺 Patient Details")
        for key, value in patient_info.items():
            st.sidebar.write(f"**{key}:** {value}")

        # Sidebar Visualization Parameters
        st.sidebar.header("Visualization Parameters")

        brain_threshold = st.sidebar.slider("Brain Threshold", 0.0, 0.5, 0.1, 0.01)
        tumor_threshold = st.sidebar.slider("Tumor Threshold", 0.3, 0.9, 0.6, 0.01)
        brain_opacity = st.sidebar.slider("Brain Opacity", 0.0, 1.0, 0.1, 0.05)
        sample_ratio = st.sidebar.slider("Sample Ratio", 0.01, 0.2, 0.05, 0.01)

        tab1, tab2 = st.tabs(["2D Slices", "3D Visualization"])

        with tab1:
            st.header("2D Slice Views")
            slice_fig = visualize_2d_slices(mri_data, brain_threshold, tumor_threshold)
            st.pyplot(slice_fig)

            st.subheader("Interactive Slice Selector")
            slice_col1, slice_col2, slice_col3 = st.columns(3)

            with slice_col1:
                x_slice = st.slider("Sagittal Slice (X)", 0, mri_data.shape[0]-1, mri_data.shape[0]//2)
                fig_x, ax_x = plt.subplots(figsize=(5, 5))
                ax_x.imshow(mri_data[x_slice, :, :].T, cmap='gray', origin='lower')
                overlay_tumor(ax_x, mri_data[x_slice, :, :].T, tumor_threshold)
                st.pyplot(fig_x)

            with slice_col2:
                y_slice = st.slider("Coronal Slice (Y)", 0, mri_data.shape[1]-1, mri_data.shape[1]//2)
                fig_y, ax_y = plt.subplots(figsize=(5, 5))
                ax_y.imshow(mri_data[:, y_slice, :].T, cmap='gray', origin='lower')
                overlay_tumor(ax_y, mri_data[:, y_slice, :].T, tumor_threshold)
                st.pyplot(fig_y)

            with slice_col3:
                z_slice = st.slider("Axial Slice (Z)", 0, mri_data.shape[2]-1, mri_data.shape[2]//2)
                fig_z, ax_z = plt.subplots(figsize=(5, 5))
                ax_z.imshow(mri_data[:, :, z_slice], cmap='gray', origin='lower')
                overlay_tumor(ax_z, mri_data[:, :, z_slice], tumor_threshold)
                st.pyplot(fig_z)

        with tab2:
            st.header("3D MRI Visualization")
            fig_3d = visualize_3d_mri(mri_data, brain_threshold, tumor_threshold, sample_ratio, 2, brain_opacity)
            st.plotly_chart(fig_3d)

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        os.unlink(tmp_file_path)