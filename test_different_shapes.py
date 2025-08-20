#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to check how pirt registration handles images with different shapes
"""

import numpy as np
import pirt
import matplotlib.pyplot as plt
from skimage import transform
import time

def test_registration_with_different_shapes():
    """Test registration with images of different shapes"""
    print("Testing registration with images of different shapes")
    
    # Create two 3D test images with different shapes
    print("Creating test images...")
    
    # First image: 100x120x80
    img1 = np.zeros((100, 120, 80), dtype=np.float32)
    # Add some basic structures
    img1[30:70, 40:80, 20:60] = 0.8  # Central cube
    img1[20:40, 30:60, 30:50] = 0.5  # Another structure
    # Add some noise
    img1 += np.random.normal(0, 0.05, img1.shape)
    img1 = np.clip(img1, 0, 1)
    
    # Second image: 90x110x75 (slightly smaller)
    img2 = np.zeros((90, 110, 75), dtype=np.float32)
    # Add similar structures but slightly shifted
    img2[25:65, 35:75, 18:58] = 0.8  # Central cube (shifted)
    img2[18:38, 28:58, 28:48] = 0.5  # Another structure (shifted)
    # Add some noise
    img2 += np.random.normal(0, 0.05, img2.shape)
    img2 = np.clip(img2, 0, 1)
    
    print(f"Image 1 shape: {img1.shape}, dtype: {img1.dtype}")
    print(f"Image 2 shape: {img2.shape}, dtype: {img2.dtype}")
    
    # Visualize middle slices of both images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(231)
    plt.imshow(img1[img1.shape[0]//2, :, :], cmap='gray')
    plt.title(f"Image 1 - Z slice")
    
    plt.subplot(232)
    plt.imshow(img1[:, img1.shape[1]//2, :], cmap='gray')
    plt.title(f"Image 1 - Y slice")
    
    plt.subplot(233)
    plt.imshow(img1[:, :, img1.shape[2]//2], cmap='gray')
    plt.title(f"Image 1 - X slice")
    
    plt.subplot(234)
    plt.imshow(img2[img2.shape[0]//2, :, :], cmap='gray')
    plt.title(f"Image 2 - Z slice")
    
    plt.subplot(235)
    plt.imshow(img2[:, img2.shape[1]//2, :], cmap='gray')
    plt.title(f"Image 2 - Y slice")
    
    plt.subplot(236)
    plt.imshow(img2[:, :, img2.shape[2]//2], cmap='gray')
    plt.title(f"Image 2 - X slice")
    
    plt.tight_layout()
    plt.savefig("test_images_different_shapes.png")
    plt.close()
    
    # Try registration
    try:
        print("Attempting registration with different shaped images...")
        start_time = time.time()
        
        # Create registration object
        reg = pirt.DiffeomorphicDemonsRegistration(img1, img2)
        
        # Set registration parameters
        reg.params.grid_sampling_factor = 1.0
        reg.params.scale_sampling = 20
        reg.params.speed_factor = 2.0
        reg.params.final_scale = 1.0
        reg.params.scale_levels = 4
        reg.params.final_grid_sampling = 1.0
        reg.params.noise_factor = 1.0
        
        # Perform registration
        reg.register(verbose=1)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Registration completed in {elapsed_time:.2f} seconds!")
        
        # Get deformation fields and transformed images
        print("Processing deformation fields...")
        deform1 = reg.get_deform(0)
        deform2 = reg.get_deform(1)
        
        transformed1 = deform1.apply_deformation(img1)
        transformed2 = deform2.apply_deformation(img2)
        
        # Convert to numpy arrays
        transformed1_array = np.asarray(transformed1)
        transformed2_array = np.asarray(transformed2)
        
        print(f"Transformed image 1 shape: {transformed1_array.shape}")
        print(f"Transformed image 2 shape: {transformed2_array.shape}")
        
        # Calculate and visualize average template
        template = np.mean([transformed1_array, transformed2_array], axis=0)
        print(f"Template shape: {template.shape}")
        
        # Visualize middle slices of transformed images and template
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(transformed1_array[transformed1_array.shape[0]//2, :, :], cmap='gray')
        plt.title(f"Transformed Image 1 - Z slice")
        
        plt.subplot(132)
        plt.imshow(transformed2_array[transformed2_array.shape[0]//2, :, :], cmap='gray')
        plt.title(f"Transformed Image 2 - Z slice")
        
        plt.subplot(133)
        plt.imshow(template[template.shape[0]//2, :, :], cmap='gray')
        plt.title(f"Template - Z slice")
        
        plt.tight_layout()
        plt.savefig("registration_results_different_shapes.png")
        plt.close()
        
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during registration: {str(e)}")
        return False
        
def test_registration_with_rescaled_images():
    """Test registration after rescaling the smaller image to match the larger one"""
    print("\nTesting registration with rescaled images...")
    
    # Create two 3D test images with different shapes
    print("Creating test images...")
    
    # First image: 100x120x80
    img1 = np.zeros((100, 120, 80), dtype=np.float32)
    # Add some basic structures
    img1[30:70, 40:80, 20:60] = 0.8  # Central cube
    img1[20:40, 30:60, 30:50] = 0.5  # Another structure
    # Add some noise
    img1 += np.random.normal(0, 0.05, img1.shape)
    img1 = np.clip(img1, 0, 1)
    
    # Second image: 90x110x75 (slightly smaller)
    img2_original = np.zeros((90, 110, 75), dtype=np.float32)
    # Add similar structures but slightly shifted
    img2_original[25:65, 35:75, 18:58] = 0.8  # Central cube (shifted)
    img2_original[18:38, 28:58, 28:48] = 0.5  # Another structure (shifted)
    # Add some noise
    img2_original += np.random.normal(0, 0.05, img2_original.shape)
    img2_original = np.clip(img2_original, 0, 1)
    
    # Resize img2 to match img1's shape
    img2 = transform.resize(img2_original, img1.shape, anti_aliasing=True, preserve_range=True)
    
    print(f"Image 1 shape: {img1.shape}, dtype: {img1.dtype}")
    print(f"Image 2 shape (after resizing): {img2.shape}, dtype: {img2.dtype}")
    
    # Try registration
    try:
        print("Attempting registration with resized images...")
        start_time = time.time()
        
        # Create registration object
        reg = pirt.DiffeomorphicDemonsRegistration(img1, img2)
        
        # Set registration parameters
        reg.params.grid_sampling_factor = 1.0
        reg.params.scale_sampling = 20
        reg.params.speed_factor = 2.0
        reg.params.final_scale = 1.0
        reg.params.scale_levels = 4
        reg.params.final_grid_sampling = 1.0
        reg.params.noise_factor = 1.0
        
        # Perform registration
        reg.register(verbose=1)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Registration completed in {elapsed_time:.2f} seconds!")
        
        # Get deformation fields and transformed images
        print("Processing deformation fields...")
        deform1 = reg.get_deform(0)
        deform2 = reg.get_deform(1)
        
        transformed1 = deform1.apply_deformation(img1)
        transformed2 = deform2.apply_deformation(img2)
        
        # Convert to numpy arrays
        transformed1_array = np.asarray(transformed1)
        transformed2_array = np.asarray(transformed2)
        
        print(f"Transformed image 1 shape: {transformed1_array.shape}")
        print(f"Transformed image 2 shape: {transformed2_array.shape}")
        
        # Calculate and visualize average template
        template = np.mean([transformed1_array, transformed2_array], axis=0)
        print(f"Template shape: {template.shape}")
        
        # Visualize middle slices of transformed images and template
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(transformed1_array[transformed1_array.shape[0]//2, :, :], cmap='gray')
        plt.title(f"Transformed Image 1 - Z slice")
        
        plt.subplot(132)
        plt.imshow(transformed2_array[transformed2_array.shape[0]//2, :, :], cmap='gray')
        plt.title(f"Transformed Image 2 - Z slice")
        
        plt.subplot(133)
        plt.imshow(template[template.shape[0]//2, :, :], cmap='gray')
        plt.title(f"Template - Z slice")
        
        plt.tight_layout()
        plt.savefig("registration_results_resized_images.png")
        plt.close()
        
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during registration: {str(e)}")
        return False

if __name__ == "__main__":
    print("==== Testing registration with different shaped images ====")
    result1 = test_registration_with_different_shapes()
    
    print("\n==== Testing registration with resized images ====")
    result2 = test_registration_with_rescaled_images()
    
    if result1 and result2:
        print("\nAll tests completed successfully!")
    else:
        print("\nSome tests failed. Check errors above.")