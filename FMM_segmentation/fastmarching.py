import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

def constrained_fast_marching(image, seeds, stopping_criteria=None):
    """
    Extended Fast Marching with multiple constraints to prevent leakage and support anisotropic growth.
    Z-level adaptive constraints are applied to handle different vessel sizes at different depths.
    
    Parameters:
    image: 3D numpy array of CT image
    seeds: List of seed points as tuples (z,y,x)
    stopping_criteria: Dict with criteria parameters
    
    Returns:
    Segmentation mask and distance map
    """
    # Default stopping criteria
    if stopping_criteria is None:
        stopping_criteria = {
            'threshold': 100,
            'max_distance': 100,
            'use_vesselness': True,
            'vesselness_weight': 0.5,
            'sigma': 1.0,
            'check_growth_rate': True,
            'growth_rate_threshold': 2.0,
            'z_scale': 1.0,                      # Growth speed along z-axis
            'xy_scale_base': 0.7,                # Base growth speed in x/y plane
            'xy_scale_z_factor': 0.01,           # Factor for z-dependent xy_scale adjustment
            'max_xy_scale': 0.8,                 # Maximum allowed xy_scale
            'min_xy_scale': 0.2,                 # Minimum allowed xy_scale
            'z_dependent_constraints': False,     # Enable z-level adaptive constraints
            'time_threshold': 50,
            'edge_weight': 0.6,                  # Weight for edge/luminosity constraints
            'vessel_size_weight': 0.5,           # Weight for vessel size constraints vs distance
            'min_vessel_radius_low_z': 2,        # Minimum vessel radius at low z-levels
            'max_vessel_radius_low_z': 5,        # Maximum vessel radius at low z-levels
            'min_vessel_radius_high_z': 4,       # Minimum vessel radius at high z-levels
            'max_vessel_radius_high_z': 10       # Maximum vessel radius at high z-levels
        }
    
    # Convert to SimpleITK
    sitk_image = sitk.GetImageFromArray(image)
    
    # Get image dimensions
    z_dim, y_dim, x_dim = image.shape
    print(f"Image dimensions: {z_dim}x{y_dim}x{x_dim}")
    
    # Create speed image (higher values = faster propagation)
    sigmoid = sitk.SigmoidImageFilter()
    sigmoid.SetOutputMinimum(0.0)
    sigmoid.SetOutputMaximum(1.0)
    sigmoid.SetAlpha(-0.5)  # Controls steepness of sigmoid
    sigmoid.SetBeta(stopping_criteria.get('threshold', 100))
    speed_image = sigmoid.Execute(sitk_image)
    
    # Check if we need to use vesselness and if Hessian filter is available
    if stopping_criteria.get('use_vesselness', True):
        try:
            # Try using traditional gradient magnitude for edge detection
            print("Using gradient magnitude for vessel enhancement...")
            gradient = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
            gradient.SetSigma(stopping_criteria.get('sigma', 1.0))
            gradient_image = gradient.Execute(sitk_image)
            
            # Invert gradient (vessels have low gradient inside, high at borders)
            inverted_gradient = sitk.BinaryNot(sitk.BinaryThreshold(
                gradient_image, 
                lowerThreshold=0.0, 
                upperThreshold=10.0, 
                insideValue=1, 
                outsideValue=0))
            
            # Combine with speed image
            weight = stopping_criteria.get('vesselness_weight', 0.5)
            # Create weight image manually (older SimpleITK doesn't have Fill)
            weight_image = sitk.Image(inverted_gradient.GetSize(), sitk.sitkFloat32)
            weight_image = sitk.AddImageFilter().Execute(weight_image, weight)
            
            weighted_gradient = sitk.Multiply(inverted_gradient, weight_image)
            speed_image = sitk.Add(speed_image, weighted_gradient)
            
        except Exception as e:
            print(f"Warning: Enhancement failed. Using basic speed image. Error: {e}")
    
    # Add edge detection to identify vessel boundaries
    # This helps prevent leakage by giving lower speed at vessel borders
    try:
        print("Adding luminosity-based edge constraints...")
        edge_weight = stopping_criteria.get('edge_weight', 0.6)
        edge_detector = sitk.CannyEdgeDetectionImageFilter()
        edge_detector.SetUpperThreshold(stopping_criteria.get('threshold', 100) + 20)
        edge_detector.SetLowerThreshold(stopping_criteria.get('threshold', 100) - 20)
        edge_detector.SetVariance(1.0)
        
        # Generate edge map
        edge_map = edge_detector.Execute(sitk_image)
        
        # Invert edge map (edges should slow down propagation)
        inverted_edge = sitk.BinaryNot(edge_map)
        
        # Create weight image for edges
        edge_weight_image = sitk.Image(inverted_edge.GetSize(), sitk.sitkFloat32)
        edge_weight_image = sitk.AddImageFilter().Execute(edge_weight_image, edge_weight)
        
        # Apply edge constraints to speed image
        weighted_edge = sitk.Multiply(inverted_edge, edge_weight_image)
        speed_image = sitk.Multiply(speed_image, weighted_edge)
    except Exception as e:
        print(f"Warning: Edge detection failed. Error: {e}")
    
    # Create a seed image
    seed_image = sitk.Image(sitk_image.GetSize(), sitk.sitkUInt8)
    seed_image.CopyInformation(sitk_image)
    
    # Place seeds
    for z, y, x in seeds:
        index = (int(x), int(y), int(z))
        try:
            seed_image[index] = 1
        except Exception as e:
            print(f"Warning: Could not place seed at {index}: {e}")
    
    # Fall back to older FastMarchingImageFilter API
    fast_marching = sitk.FastMarchingImageFilter()
    
    # Set initial seeds directly (older API)
    print("Adding seed points for Fast Marching...")
    trial_points_added = 0
    for z, y, x in seeds:
        if 0 <= z < image.shape[0] and 0 <= y < image.shape[1] and 0 <= x < image.shape[2]:
            # Set trial points in SimpleITK's x,y,z order
            try:
                fast_marching.AddTrialPoint([int(x), int(y), int(z)])
                trial_points_added += 1
            except Exception as e:
                print(f"Warning: Could not add trial point at ({x},{y},{z}): {e}")
    
    print(f"Added {trial_points_added} trial points for Fast Marching")
    
    # Set stopping value
    fast_marching.SetStoppingValue(stopping_criteria.get('max_distance', 100))
    
    # Apply z-level adaptive constraints if enabled
    if stopping_criteria.get('z_dependent_constraints', True):
        # Create an anisotropy image that varies with z-level
        print("Creating z-dependent anisotropy map...")
        anisotropy_np = np.zeros((z_dim, y_dim, x_dim), dtype=np.float32)
        
        # Define z-level dependent xy-scale
        xy_scale_base = stopping_criteria.get('xy_scale_base', 0.3)
        xy_scale_z_factor = stopping_criteria.get('xy_scale_z_factor', 0.01)
        max_xy_scale = stopping_criteria.get('max_xy_scale', 0.8)
        min_xy_scale = stopping_criteria.get('min_xy_scale', 0.2)
        
        # Calculate z-dependent scaling for each level
        for z in range(z_dim):
            # Normalized z-position [0,1]
            z_norm = z / max(1, z_dim - 1)
            
            # Calculate xy_scale: increases with z-level
            # Higher z = larger vessels = more lenient xy constraints
            xy_scale = xy_scale_base + (z_norm * xy_scale_z_factor * z_dim)
            xy_scale = min(max_xy_scale, max(min_xy_scale, xy_scale))
            
            # Apply to this z-level
            anisotropy_np[z, :, :] = xy_scale
        
        # Create spacing image
        anisotropy_image = sitk.GetImageFromArray(anisotropy_np)
        anisotropy_image.CopyInformation(sitk_image)
        
        # Apply anisotropy to the sitk image
        z_scale = stopping_criteria.get('z_scale', 1.0)
        sitk_image.SetSpacing((1.0, 1.0, z_scale))
        
        # Debug info
        print(f"Z-scale: {z_scale}, XY-scale range: {min_xy_scale}-{max_xy_scale}")
    else:
        # Set fixed anisotropic weights if specified
        z_scale = stopping_criteria.get('z_scale', 1.0)
        xy_scale = stopping_criteria.get('xy_scale_base', 0.3)
        spacing = (xy_scale, xy_scale, z_scale)
        sitk_image.SetSpacing(spacing)
        seed_image.SetSpacing(spacing)
        print(f"Using fixed anisotropy: Z-scale={z_scale}, XY-scale={xy_scale}")
    
    # Run fast marching
    try:
        print("Running Fast Marching...")
        distance_map = fast_marching.Execute(speed_image)
        
        # Threshold distance map to get segmentation
        print("Thresholding distance map...")
        threshold_filter = sitk.BinaryThresholdImageFilter()
        threshold_filter.SetLowerThreshold(0)
        threshold_filter.SetUpperThreshold(stopping_criteria.get('time_threshold', 50))
        threshold_filter.SetInsideValue(1)
        threshold_filter.SetOutsideValue(0)
        segmentation_image = threshold_filter.Execute(distance_map)
        
    except Exception as e:
        print(f"Fast Marching error: {e}. Falling back to region growing...")
        # Fall back to connected threshold or region growing
        connected = sitk.ConnectedThresholdImageFilter()
        for z, y, x in seeds:
            if 0 <= z < image.shape[0] and 0 <= y < image.shape[1] and 0 <= x < image.shape[2]:
                seed_value = image[z, y, x]
                # Add some flexibility to the thresholds for better segmentation
                lower = max(0, seed_value - 25)
                upper = min(255, seed_value + 25)
                
                # Add seed in x,y,z order for SimpleITK
                try:
                    connected.AddSeed([int(x), int(y), int(z)])
                except Exception as seed_err:
                    print(f"Warning: Could not add seed: {seed_err}")
        
        # Create custom thresholds for better vessel segmentation
        lower_threshold = stopping_criteria.get('threshold', 100) - 50
        upper_threshold = stopping_criteria.get('threshold', 100) + 150
        connected.SetLower(lower_threshold)
        connected.SetUpper(upper_threshold)
        
        try:
            segmentation_image = connected.Execute(sitk_image)
        except Exception as conn_err:
            print(f"Region growing also failed: {conn_err}")
            # Worst case: create empty segmentation
            segmentation_image = sitk.Image(sitk_image.GetSize(), sitk.sitkUInt8)
        
        # Create distance map from segmentation
        try:
            distance_map = sitk.SignedMaurerDistanceMapImageFilter().Execute(segmentation_image)
        except Exception:
            # If distance map fails, use binary image as placeholder
            distance_map = segmentation_image
    
    # Convert back to numpy arrays
    segmentation = sitk.GetArrayFromImage(segmentation_image).astype(np.uint8)
    distance_map_np = sitk.GetArrayFromImage(distance_map)
    
    # Apply z-level dependent vessel size constraints
    if stopping_criteria.get('z_dependent_constraints', True):
        print("Applying z-level dependent vessel size constraints...")
        try:
            # Parameters for vessel size constraints
            min_radius_low_z = stopping_criteria.get('min_vessel_radius_low_z', 2)
            max_radius_low_z = stopping_criteria.get('max_vessel_radius_low_z', 5)
            min_radius_high_z = stopping_criteria.get('min_vessel_radius_high_z', 4)
            max_radius_high_z = stopping_criteria.get('max_vessel_radius_high_z', 10)
            
            # Create refined segmentation based on z-level vessel size constraints
            refined_segmentation = np.zeros_like(segmentation)
            
            # Label connected components in the segmentation
            labeled, num_components = ndimage.label(segmentation)
            
            # Process each connected component
            for label in range(1, num_components + 1):
                component = (labeled == label)
                
                # Get z-range of the component
                z_indices = np.where(np.any(component, axis=(1, 2)))[0]
                if len(z_indices) == 0:
                    continue
                    
                z_min, z_max = z_indices.min(), z_indices.max()
                z_center = (z_min + z_max) // 2
                z_norm = z_center / max(1, z_dim - 1)  # Normalized z-position [0,1]
                
                # Interpolate vessel size constraints based on z-level
                min_radius = min_radius_low_z + z_norm * (min_radius_high_z - min_radius_low_z)
                max_radius = max_radius_low_z + z_norm * (max_radius_high_z - max_radius_low_z)
                
                # For each z-slice, check vessel size in xy-plane
                for z in z_indices:
                    component_slice = component[z]
                    if not np.any(component_slice):
                        continue
                        
                    # Calculate center of mass in this slice
                    y_indices, x_indices = np.where(component_slice)
                    center_y = np.mean(y_indices)
                    center_x = np.mean(x_indices)
                    
                    # Check distance from center to edge (approximate radius)
                    distances = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
                    max_distance = np.max(distances)
                    
                    # If the vessel radius is within acceptable range, keep it
                    if min_radius <= max_distance <= max_radius:
                        refined_segmentation[z, component_slice] = 1
                    else:
                        # If vessel is too large but in a high z-level, we might accept it
                        # based on intensity consistency
                        if max_distance > max_radius and z_norm > 0.7:
                            intensity_std = np.std(image[z][component_slice])
                            if intensity_std < 15:  # Low variance indicates true vessel
                                refined_segmentation[z, component_slice] = 1
            
            # Update segmentation with size-constrained version
            if np.any(refined_segmentation):
                segmentation = refined_segmentation
                print(f"Applied vessel size constraints. Remaining voxels: {np.sum(segmentation)}")
            else:
                print("Warning: Size constraints removed all vessels. Keeping original segmentation.")
        except Exception as size_err:
            print(f"Error applying vessel size constraints: {size_err}")
    
    # Additional post-processing if needed
    if stopping_criteria.get('check_growth_rate', True):
        print("Post-processing segmentation...")
        try:
            # Clean up small islands
            labeled, num_features = ndimage.label(segmentation)
            if num_features > 0:
                sizes = ndimage.sum(segmentation, labeled, range(1, num_features + 1))
                min_size = 50  # Minimum size to keep
                mask = np.zeros_like(segmentation, dtype=bool)
                for i, size in enumerate(sizes):
                    if size >= min_size:
                        mask[labeled == i + 1] = True
                segmentation = mask.astype(np.uint8)
                
                # Additional connectivity enhancement: 
                # Try to reconnect vessels that may have been fragmented
                segmentation = ndimage.binary_closing(segmentation, 
                                                     structure=np.ones((3,3,3))).astype(np.uint8)
        except Exception as post_err:
            print(f"Post-processing error: {post_err}")
    
    # Final enhancement: ensure vessels maintain connectivity along z-axis
    try:
        print("Enhancing vessel connectivity...")
        # Use dilation and morphological operations to ensure connectivity
        connected_vessels = np.zeros_like(segmentation)
        
        # Create a small bridge in the z direction
        bridge = np.zeros((3,3,3), dtype=np.uint8)
        bridge[0,1,1] = bridge[1,1,1] = bridge[2,1,1] = 1
        
        # Apply closing operation with z-oriented kernel to maintain connectivity
        connected_vessels = ndimage.binary_closing(segmentation, structure=bridge).astype(np.uint8)
        
        # Only update if we maintain a reasonable size
        if np.sum(connected_vessels) < np.sum(segmentation) * 1.5:
            segmentation = connected_vessels
    except Exception as e:
        print(f"Connectivity enhancement error: {e}")
    
    print(f"Segmentation complete. Found {np.sum(segmentation)} vessel voxels")
    return segmentation, distance_map_np