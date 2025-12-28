"""
Gaussian Mixture Model-based habitat clustering for radiomics feature analysis
"""
import os
import glob
import numpy as np
import SimpleITK as sitk
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy import ndimage
from config import CLUSTERING_PARAMS


class HabitatClusterAnalyzer:
    """Performs habitat clustering using GMM on radiomics features"""
    
    def __init__(self, n_habitats=None):
        """
        Initialize habitat analyzer
        
        Parameters:
        -----------
        n_habitats : int, optional
            Number of habitats to identify (default from config)
        """
        self.K = n_habitats or CLUSTERING_PARAMS['K_HABITAT']
        self.EPS = CLUSTERING_PARAMS['EPS']
        self.normalization_method = CLUSTERING_PARAMS['normalization_method']
    
    def load_feature_maps(self, feature_dir, roi_mask):
        """
        Load feature maps and organize into tensor
        
        Parameters:
        -----------
        feature_dir : str
            Directory containing feature map NIfTI files
        roi_mask : np.ndarray
            Binary mask of region of interest
            
        Returns:
        --------
        tuple : (feature_tensor, reference_image, feature_names)
        """
        # Find all feature map files
        feature_files = sorted(glob.glob(os.path.join(feature_dir, "feature_map_*.nii.gz")))
        if len(feature_files) == 0:
            raise FileNotFoundError(f"No feature maps found in {feature_dir}")
        
        maps = []
        feat_names = []
        
        for f in feature_files:
            # Load feature map
            img = sitk.ReadImage(f)
            arr = sitk.GetArrayFromImage(img)  # Shape: [Depth, Height, Width]
            arr = arr * roi_mask  # Mask outside ROI
            
            maps.append(arr)
            feat_names.append(os.path.basename(f))
        
        # Stack into tensor: [Channels, Depth, Height, Width]
        X_maps = np.stack(maps, axis=0)
        
        return X_maps, img, feat_names
    
    def normalize_features(self, X_roi):
        """
        Normalize features using specified method
        
        Parameters:
        -----------
        X_roi : np.ndarray
            Feature matrix [n_voxels, n_features]
            
        Returns:
        --------
        np.ndarray : Normalized feature matrix
        """
        if self.normalization_method == 'zscore':
            # Z-score normalization
            mean = X_roi.mean(axis=0)
            std = X_roi.std(axis=0) + 1e-6
            return (X_roi - mean) / std
        
        elif self.normalization_method == 'mad':
            # Median Absolute Deviation normalization (robust to outliers)
            median = np.median(X_roi, axis=0)
            mad = np.median(np.abs(X_roi - median), axis=0) + 1e-6
            return (X_roi - median) / mad
        
        else:
            # No normalization
            return X_roi
    
    def compute_habitat_diversity_metrics(self, prob_roi):
        """
        Compute habitat diversity and complexity metrics
        
        Parameters:
        -----------
        prob_roi : np.ndarray
            Habitat probabilities [n_voxels, K]
            
        Returns:
        --------
        dict : Dictionary of habitat metrics
        """
        metrics = {}
        
        # 1. Habitat composition ratios
        ratios = prob_roi.mean(axis=0)
        for i in range(self.K):
            metrics[f"habitat_{i+1}_ratio"] = ratios[i]
        
        # 2. Voxel-level entropy (uncertainty in habitat assignment)
        voxel_entropy = -np.sum(prob_roi * np.log(prob_roi + self.EPS), axis=1)
        metrics["habitat_entropy_mean"] = voxel_entropy.mean()
        metrics["habitat_entropy_std"] = voxel_entropy.std()
        
        # 3. Probability variance (habitat mixing)
        prob_var = np.var(prob_roi, axis=1)
        metrics["habitat_prob_variance_mean"] = prob_var.mean()
        metrics["habitat_prob_variance_std"] = prob_var.std()
        
        # 4. Habitat dominance (most prevalent habitat)
        mean_probs = prob_roi.mean(axis=0)
        metrics["habitat_dominance_ratio"] = mean_probs.max()
        
        # 5. Effective number of habitats (diversity index)
        metrics["habitat_num_effective"] = 1.0 / np.sum(mean_probs**2)
        
        # 6. Soft volume entropy (distribution evenness)
        metrics["habitat_soft_volume_entropy"] = -np.sum(
            mean_probs * np.log(mean_probs + self.EPS)
        )
        
        return metrics
    
    def compute_spatial_metrics(self, habitat_maps, roi_mask):
        """
        Compute spatial characteristics of habitats
        
        Parameters:
        -----------
        habitat_maps : np.ndarray
            Habitat probability maps [K, Depth, Height, Width]
        roi_mask : np.ndarray
            Binary ROI mask
            
        Returns:
        --------
        dict : Spatial metrics dictionary
        """
        metrics = {}
        edge_values = []
        
        for k in range(self.K):
            habitat_map = habitat_maps[k]  # Soft probability map for habitat k
            
            # Compute 3D gradient magnitude (edge strength)
            grad_x = np.gradient(habitat_map, axis=0)
            grad_y = np.gradient(habitat_map, axis=1)
            grad_z = np.gradient(habitat_map, axis=2)
            
            grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
            
            # Average gradient within ROI (higher = more boundaries)
            edge_ratio = grad_mag[roi_mask].mean()
            edge_values.append(edge_ratio)
            metrics[f"habitat_edge_ratio_{k+1}"] = edge_ratio
        
        # Summary edge statistics
        metrics["habitat_edge_mean"] = np.mean(edge_values)
        metrics["habitat_edge_std"] = np.std(edge_values)
        
        return metrics
    
    def analyze_modality(self, feature_dir, roi_mask, output_dir):
        """
        Perform habitat analysis for a single imaging modality
        
        Parameters:
        -----------
        feature_dir : str
            Directory with feature maps for this modality
        roi_mask : np.ndarray
            Binary ROI mask
        output_dir : str
            Output directory for habitat maps
            
        Returns:
        --------
        dict : Comprehensive habitat metrics
        """
        # Load and prepare feature data
        X_maps, ref_img, _ = self.load_feature_maps(feature_dir, roi_mask)
        spatial_shape = X_maps.shape[1:]  # Original 3D shape
        
        # Reshape to voxel × feature matrix
        C = X_maps.shape[0]  # Number of features
        X = X_maps.reshape(C, -1).T  # [n_voxels, n_features]
        roi_flat = roi_mask.flatten()
        X_roi = X[roi_flat, :]  # Features only within ROI
        
        print(f"Processing {X_roi.shape[0]} voxels with {X_roi.shape[1]} features")
        
        # Normalize features
        X_roi_norm = self.normalize_features(X_roi)
        
        # Fit Gaussian Mixture Model
        print(f"Fitting GMM with {self.K} components...")
        gmm = GaussianMixture(n_components=self.K, random_state=42, covariance_type='full')
        gmm.fit(X_roi_norm)
        
        # Get soft habitat assignments (probabilities)
        prob_roi = gmm.predict_proba(X_roi_norm)  # [n_voxels, K]
        
        # Reconstruct full 3D probability maps
        prob_full = np.zeros((roi_flat.shape[0], self.K), dtype=np.float32)
        prob_full[roi_flat] = prob_roi
        habitat_maps = prob_full.T.reshape((self.K,) + spatial_shape)
        
        # Save habitat probability maps as NIfTI
        os.makedirs(output_dir, exist_ok=True)
        for k in range(self.K):
            habitat_img = sitk.GetImageFromArray(habitat_maps[k])
            habitat_img.CopyInformation(ref_img)  # Preserve spatial metadata
            sitk.WriteImage(habitat_img, os.path.join(output_dir, f"habitat_{k+1}.nii.gz"))
        
        # Compute comprehensive metrics
        print("Computing habitat metrics...")
        metrics = self.compute_habitat_diversity_metrics(prob_roi)
        spatial_metrics = self.compute_spatial_metrics(habitat_maps, roi_mask)
        
        # Combine all metrics
        metrics.update(spatial_metrics)
        
        return metrics