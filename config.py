"""
Centralized configuration management for glioma habitat analysis
"""

# ==================== PATH CONFIGURATION ====================
BASE_PATHS = {
    'T2': r'\glioma_T2_TRA',
    'T2_FLAIR': r'\glioma_T2_FLAIR_TRA', 
    'T1CE': r'\glioma_T1CE_TRA',
    'ADC': r'\glioma_ADC_TRA',
    'CBF': r'\glioma_CBF_TRA',
    'mask': r'\glioma_masks',
    'output': r'\glioma_output'
}

# ==================== FEATURE EXTRACTION ====================
FEATURE_SET = [
    'original_firstorder_Mean', 'original_firstorder_Std', 'original_firstorder_Skewness',
    'original_firstorder_Kurtosis', 'original_firstorder_Entropy',
    'original_glcm_Energy', 'original_glcm_Contrast', 'original_glcm_Correlation',
    'original_glcm_Homogeneity', 'original_glcm_Dissimilarity',
    'original_glcm_Autocorrelation', 'original_glcm_ClusterShade',
    'original_glcm_ClusterProminence', 'original_glcm_Imc1', 'original_glcm_Imc2',
    'original_glcm_MaximumProbability', 'original_glrlm_ShortRunEmphasis',
    'original_glrlm_LongRunEmphasis', 'original_glrlm_GreyLevelNonUniformity',
    'original_glrlm_RunLengthNonUniformity', 'original_glrlm_RunPercentage',
    'original_glrlm_LowGreyLevelRunEmphasis', 'original_glrlm_HighGreyLevelRunEmphasis',
    'original_glszm_SmallAreaEmphasis', 'original_glszm_LargeAreaEmphasis',
    'original_glszm_ZonePercentage', 'original_glszm_GreyLevelNonUniformity',
    'original_glszm_SizeZoneNonUniformity', 'original_glszm_LowGreyLevelZoneEmphasis',
    'original_glszm_HighGreyLevelZoneEmphasis', 'original_ngtdm_Coarseness',
    'original_ngtdm_Contrast', 'original_ngtdm_Busyness', 'original_ngtdm_Complexity',
    'original_ngtdm_Strength'
]

SLIDING_WINDOW_PARAMS = {
    'window_size': 7,           # Size of sliding window (odd number)
    'min_valid_voxels': 3,      # Minimum valid voxels in window to compute features
    'binWidth': 25              # Bin width for intensity discretization
}

# ==================== CLUSTERING PARAMETERS ====================
CLUSTERING_PARAMS = {
    'K_HABITAT': 3,             # Number of habitats to identify
    'EPS': 1e-8,                # Small epsilon to avoid log(0)
    'normalization_method': 'mad'  # 'zscore' or 'mad' (median absolute deviation)
}

# ==================== PROCESSING CONFIG ====================
PROCESSING = {
    'modalities': ['ADC', 'DWI', 'T2'],  # Imaging modalities to process
    'num_workers': -1                     # -1 means use all CPU cores minus one
}