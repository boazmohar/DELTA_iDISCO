import itk
import os
from DELAT_utils import read_h5_image, flip_image

# Combined function for registration and applying transforms
def register_and_transform(fx, files, output_dir, param_files, logger, flip_axes=None):
    num_cores = os.cpu_count()  
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(num_cores)
    logger.info(f"Register_and_transform started, set ITK to use {num_cores} threads.")

    # Step 1: Register ch0 (moving image) with the fixed image
    mv = read_h5_image(files['ch0'], 'Data')
    
    # Flip the image if flip_axes is not empty
    if flip_axes:
        mv = flip_image(mv, axes=flip_axes)
        logger.info(f"Flipped image {files['ch0']} along axes {flip_axes}")
    logger.info(f"Moving image read from  {files['ch0']}")
    # Create a new ParameterObject for registration
    parameter_object = itk.ParameterObject.New()
    for p in param_files:
        parameter_object.AddParameterFile(p)
    
    logger.info("param files read")
    # Create output directory if it doesn't exist
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory {output_dir}")
    
    # Perform registration (this will create TransformParameters.0.txt, etc.)
    res, params = itk.elastix_registration_method(fx, mv, parameter_object, log_to_file=True, output_directory=output_dir)
    logger.info(f"Registration done for {output_dir}")
    
    # Step 2: Apply the generated transforms to all channels (ch0, ch1, ch2)
    param_files_generated = [f'TransformParameters.{i}.txt' for i in range(4)]
    
    # Create a new ParameterObject for transformix
    parameter_object_transformix = itk.ParameterObject.New()
    for p in param_files_generated:
        parameter_path = os.path.join(output_dir, p)
        if os.path.exists(parameter_path):
            parameter_object_transformix.AddParameterFile(parameter_path)
        else:
            logger.warning(f"Warning: Parameter file {parameter_path} not found.")
    
    # Apply the transform for each channel
    for name, path in files.items():
        logger.info(f'Applying transform to {name}')
        moving_image = read_h5_image(path, 'Data')  # Read the image
        
        # Flip the image if flip_axes is not empty
        if flip_axes:
            moving_image = flip_image(moving_image, axes=flip_axes)
            logger.info(f"Flipped image {path} along axes {flip_axes}")
            
        itk_image = itk.image_from_array(moving_image)
        logger.info(f'Moving_image from {path} loaded and converted to itk')
        transformix_filter = itk.TransformixFilter.New(Input=itk_image, TransformParameterObject=parameter_object_transformix)
        transformix_filter.SetComputeSpatialJacobian(False)
        transformix_filter.SetComputeDeterminantOfSpatialJacobian(False)
        transformix_filter.SetComputeDeformationField(False)
        transformix_filter.Update()
        
        # Save the transformed image
        transformed_image = transformix_filter.GetOutput()
        output_image_path = os.path.join(output_dir, name + '.tif')
        itk.imwrite(transformed_image, output_image_path)
        logger.info(f"Transformed image saved to {output_image_path}")