import itk
import os
from .utils import read_h5_image

# Combined function for registration and applying transforms
def register_and_transform(fx, files, output_dir, param_files):
    # Step 1: Register ch0 (moving image) with the fixed image
    mv = read_h5_image(files['ch0'], 'Data')
    
    # Create a new ParameterObject for registration
    parameter_object = itk.ParameterObject.New()
    for p in param_files:
        parameter_object.AddParameterFile(p)
    
    # Create output directory if it doesn't exist
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Perform registration (this will create TransformParameters.0.txt, etc.)
    res, params = itk.elastix_registration_method(fx, mv, parameter_object, log_to_file=True, output_directory=output_dir)
    print(f"Registration done for {output_dir}")
    
    # Step 2: Apply the generated transforms to all channels (ch0, ch1, ch2)
    # The registration step should generate TransformParameters.{i}.txt in the output directory.
    param_files_generated = [f'TransformParameters.{i}.txt' for i in range(4)]
    
    # Create a new ParameterObject for transformix
    parameter_object_transformix = itk.ParameterObject.New()
    for p in param_files_generated:
        parameter_path = os.path.join(output_dir, p)
        if os.path.exists(parameter_path):
            parameter_object_transformix.AddParameterFile(parameter_path)
        else:
            print(f"Warning: Parameter file {parameter_path} not found.")
    
    # Apply the transform for each channel
    for name, path in files.items():
        print(f'Applying transform to {name}')
        moving_image = read_h5_image(path, 'Data')  # Read the image
        transformix_filter = itk.TransformixFilter.New(Input=moving_image, TransformParameterObject=parameter_object_transformix)
        transformix_filter.SetComputeSpatialJacobian(False)
        transformix_filter.SetComputeDeterminantOfSpatialJacobian(False)
        transformix_filter.SetComputeDeformationField(False)
        transformix_filter.Update()
        
        # Save the transformed image
        transformed_image = transformix_filter.GetOutput()
        output_image_path = os.path.join(output_dir, name + '.tif')
        itk.imwrite(transformed_image, output_image_path)
        print(f"Transformed image saved to {output_image_path}")
