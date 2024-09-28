import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.ndimage import zoom, sobel
import plotly.io as pio
pio.renderers.default = 'iframe'  # Set to 'notebook' for JupyterLab support
from skimage import measure


def downsample_volume(volume, downsample_factor):
    """
    Downsample a 3D volume by a specified factor using zoom.
    
    Parameters:
        volume (ndarray): The input 3D volume to downsample.
        downsample_factor (float): Factor by which to downsample the volume.
    
    Returns:
        downsampled_volume (ndarray): The downsampled 3D volume.
    """
    return zoom(volume, (1/downsample_factor, 1/downsample_factor, 1/downsample_factor), order=0)

# Function to create a scale bar along X, Y, Z axes
def add_scale_bars(fig, length=1, offset=0):
    # X-axis scale bar
    fig.add_trace(go.Scatter3d(
        x=[offset, offset + length],
        y=[0, 0],
        z=[0, 0],
        mode='lines',
        line=dict(color='red', width=5),
        showlegend=False 
    ))

    # Y-axis scale bar
    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[offset, offset + length],
        z=[0, 0],
        mode='lines',
        line=dict(color='green', width=5),
        showlegend=False 
    ))

    # Z-axis scale bar
    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[offset, offset + length],
        mode='lines',
        line=dict(color='blue', width=5),
        showlegend=False 
    ))


# Plotting function reuses the precomputed vertices and faces
def plot_interactive_3d_brain_isosurface(region_ids, annotation_data, verts, faces, 
                                         ccf_resolution=0.05, region_names=None,
                                        percentage_changes=None):
    # Create the plotly figure
    fig = go.Figure()

    # Mask and region extraction
    masks = [np.isin(annotation_data, region_id) for region_id in region_ids]
    # Get the indices of the masked regions for each region
    region_coords = [np.column_stack(np.where(mask)) for mask in masks]
    
    # Convert voxel indices to real-world coordinates (in millimeters)
    region_coords = [coords * ccf_resolution for coords in region_coords]

    # Define colors for the regions
    colors = ['red', 'blue', 'green', 'orange', 'purple']  
   
    # Plot the regions with ordered colors based on importance
    for i, coords in enumerate(region_coords):
        if not percentage_changes is None:
            label = f"{region_names[i]} ({percentage_changes[i]:.1f}% change)"
        else:
            label = region_names[i] if region_names else f"Region {region_ids[i]}"

        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(size=6, color=colors[i], opacity=0.8),
            name=label
        ))

    # Convert voxel indices to real-world coordinates in millimeters for the brain outline
    verts = verts * ccf_resolution
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    # Plot the 3D surface for the brain outline using the precomputed marching cubes vertices and faces
    fig.add_trace(go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.3,  # Adjust transparency for the outline
        color='gray',
        name='Brain Isosurface'
    ))
    camera  = dict(
    eye=dict(x=-.5, y=-2.5, z=-1),    # Camera placed diagonally above, right, and front
    center=dict(x=0, y=0, z=0),
    up=dict(x=0, y=0, z=1)      # Z-axis is up
)
    # Update layout for better viewing and set to mm
    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(x=0, y=1),
        scene_camera=camera
    )

    # Add custom scale bars
    add_scale_bars(fig)
    
    # Hide the default background axes
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showticklabels=False, visible=False)
        )
    )
    # Set the iframe renderer for JupyterLab
    pio.renderers.default = 'iframe'
    fig.show()
    return fig
    
def get_values(df):
    region_names = df['name'].tolist()
    region_ids = df['id'].tolist()
    change = (df['ratio'].values-1)*100
    return region_ids, region_names, change
    