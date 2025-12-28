"""
Sliding-window radiomics feature extraction module for voxel-wise habitat analysis
"""
import os
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from config import FEATURE_SET, SLIDING_WINDOW_PARAMS


class RadiomicsFeatureExtractor:
    """Extracts radiomics features using sliding window approach"""
    
    def __init__(self):
        self.features_to_extract = FEATURE_SET
        self.window_size = SLIDING_WINDOW_PARAMS['window_size']
        self.min_valid_voxels = SLIDING_WINDOW_PARAMS['min_valid_voxels']
        
        # Configure PyRadiomics feature extractor
        self.params = {
            'binWidth': SLIDING_WINDOW_PARAMS['binWidth'],
            'enableCExtensions': True,      # Use C extensions for speed
            'enableAllFeatures': False,      # Only extract enabled features
            'featureClass': ['glcm', 'firstorder', 'glrlm', 'glszm', 'ngtdm'],
            'enabledFeatures': self._categorize_features()
        }
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**self.params)
    
    def _categorize_features(self):
        """Organize features by their class for PyRadiomics configuration"""
        enabled_features = {
            'glcm': [], 'firstorder': [], 'glrlm': [], 'glszm': [], 'ngtdm': []
        }
        
        for feat in self.features_to_extract:
            # Extract feature class and name from feature string
            feat_class = feat.split('_')[1]   # e.g., 'glcm', 'firstorder'
            feat_name = feat.split('_')[-1]    # actual feature name
            if feat_class in enabled_features:
                enabled_features[feat_class].append(feat_name)
        
        return enabled_features
    
    def sliding_window_feature_maps(self, image_sitk, mask_sitk):
        """
        Generate feature maps using sliding window approach
        
        Parameters:
        -----------
        image_sitk : SimpleITK.Image
            Input medical image
        mask_sitk : SimpleITK.Image
            Binary mask defining ROI
            
        Returns:
        --------
        dict : Dictionary of feature maps {feature_name: feature_map_array}
        """
        # Convert SimpleITK images to numpy arrays
        img_arr = sitk.GetArrayFromImage(image_sitk)
        mask_arr = sitk.GetArrayFromImage(mask_sitk)
        radius = self.window_size // 2  # Window radius
        
        # Initialize feature maps with zeros
        feature_maps_dict = {
            feat: np.zeros_like(img_arr, dtype=np.float32) 
            for feat in self.features_to_extract
        }
        
        # Get all voxel indices within ROI
        indices = np.argwhere(mask_arr > 0)
        
        print(f"Processing {len(indices)} voxels in ROI...")
        
        for idx in indices:
            # Define sliding window bounds for each dimension
            slices = [
                slice(max(i - radius, 0), min(i + radius + 1, img_arr.shape[d]))
                for d, i in enumerate(idx)
            ]
            
            # Extract window sub-volumes
            window_img = img_arr[tuple(slices)]
            window_mask = mask_arr[tuple(slices)]
            
            # Skip windows with insufficient valid voxels
            if np.sum(window_mask) < self.min_valid_voxels:
                continue
            
            # Convert window to SimpleITK for feature extraction
            window_img_sitk = sitk.GetImageFromArray(window_img)
            window_mask_sitk = sitk.GetImageFromArray(window_mask)
            
            try:
                # Extract features from current window
                features = self.extractor.execute(window_img_sitk, window_mask_sitk)
            except Exception as e:
                # Skip voxel if feature extraction fails
                continue
            
            # Populate feature maps
            for feat in self.features_to_extract:
                feature_maps_dict[feat][tuple(idx)] = features.get(feat, 0.0)
        
        return feature_maps_dict
    
    def process_image_pair(self, image_path, mask_path, output_dir, patient_id, modality):
        """
        Process single image-mask pair and save feature maps
        
        Parameters:
        -----------
        image_path : str
            Path to input image
        mask_path : str
            Path to mask file
        output_dir : str
            Base output directory
        patient_id : str
            Patient identifier
        modality : str
            Imaging modality (T2, ADC, DWI)
            
        Returns:
        --------
        str : Status message
        """
        # Validate file existence
        if not os.path.exists(image_path):
            return f"Image not found: {image_path}"
        if not os.path.exists(mask_path):
            return f"Mask not found: {mask_path}"
        
        # Load images
        image_sitk = sitk.ReadImage(image_path)
        mask_sitk = sitk.ReadImage(mask_path)
        
        # Extract feature maps
        feature_maps = self.sliding_window_feature_maps(image_sitk, mask_sitk)
        
        # Create output directory
        out_dir = os.path.join(output_dir, patient_id, modality)
        os.makedirs(out_dir, exist_ok=True)
        
        # Save each feature map as NIfTI file
        for feat_name, fmap in feature_maps.items():
            fmap_sitk = sitk.GetImageFromArray(fmap)
            fmap_sitk.CopyInformation(image_sitk)  # Preserve spatial metadata
            
            output_path = os.path.join(out_dir, f'feature_map_{feat_name}.nii.gz')
            sitk.WriteImage(fmap_sitk, output_path)
        
        return f"Extracted {len(feature_maps)} features for {patient_id}/{modality}"