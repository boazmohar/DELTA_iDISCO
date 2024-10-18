import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import os
import itk
from tifffile import imread

# Generate projections
def generate_projections(data):
    z_projection = np.max(data, axis=0)
    xz_projection = np.max(data, axis=2)
    return z_projection, xz_projection

def normalize_and_convert(data):
    p_low, p_high = np.percentile(data, (0.1, 99))
    data = np.clip(data, p_low, p_high)
    data = (data - p_low) / (p_high - p_low) * 255
    return data.astype(np.uint8)

# Combine projections into RGB images
def combine_projections(atlas_proj, channel_proj, atlas_views, channel_views):
    x = normalize_and_convert(channel_proj[0])
    combined_z = np.stack((x,atlas_proj[0],x), axis=-1)
    x = normalize_and_convert(channel_proj[1])
    combined_xz =  np.stack((x,atlas_proj[1],x), axis=-1)
    x = normalize_and_convert(channel_views[0])
    combined_view1 = np.stack((x,atlas_views[0],x), axis=-1)
    x = normalize_and_convert(channel_views[1])
    combined_view2 = np.stack((x,atlas_views[1],x), axis=-1)
    return combined_z, combined_xz, combined_view1, combined_view2

def create_pdf_report(animal_id, projections, output_dir):
    pdf_path = os.path.join(output_dir, f"{animal_id}_QC_report.pdf")
    with PdfPages(pdf_path) as pdf:
        for proj in projections:
            plt.figure()
            plt.imshow(proj)
            plt.title(f"{animal_id} Projection")
            pdf.savefig()
            plt.close()

def generate_views(data, x_location=400, y_location=500):
    view1 = data[x_location, :, :]
    view2 = data[:, y_location, :]
    return view1, view2

# Main function
def make_qc(base_path, atlas_path, csv_paths):

    # Load atlas data
    itk_atlas = itk.imread(atlas_path, itk.US)
    atlas_data = itk.array_view_from_image(itk_atlas)
    atlas_projections = generate_projections(atlas_data)
    atlas_views = generate_views(atlas_data)

    for anm, csv_path in csv_paths.items():
        # Load CSV file
        pdf_path = os.path.join(base_path, f"{anm}_QC_report.pdf")
        if os.path.exists(pdf_path):
            print(f"PDF report for {anm} already exists. Skipping...")
            continue
        csv_dir = os.path.dirname(csv_path)
        print(f"Processing {anm} at {csv_dir}")
        ch0_path = os.path.join(csv_dir, 'ch0.tif')
        ch1_path = os.path.join(csv_dir, 'ch1.tif')
        ch2_path = os.path.join(csv_dir, 'ch2.tif')

        ch0_data = imread(ch0_path)
        ch1_data = imread(ch1_path)
        ch2_data = imread(ch2_path)
        print('read data')
        # Generate projections
        ch0_projections = generate_projections(ch0_data)
        ch1_projections = generate_projections(ch1_data)
        ch2_projections = generate_projections(ch2_data)
        print('generated projections')
        ch0_views = generate_views(ch0_data)
        ch1_views = generate_views(ch1_data)
        ch2_views = generate_views(ch2_data)
        print('generated views')
        # Combine projections into RGB images
        combined_projections_ch0 = combine_projections(atlas_projections, ch0_projections, atlas_views, ch0_views)
        combined_projections_ch1 = combine_projections(atlas_projections, ch1_projections, atlas_views, ch1_views)
        combined_projections_ch2 = combine_projections(atlas_projections, ch2_projections, atlas_views, ch2_views)
        print('combined projections')
        # Create PDF report
        create_pdf_report(anm, combined_projections_ch0 + combined_projections_ch1 + combined_projections_ch2, base_path)
        print(f"PDF report for {anm} created successfully.")
