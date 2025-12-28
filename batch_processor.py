"""
Main batch processing pipeline for glioma habitat analysis
"""
import os
import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk

from config import BASE_PATHS, PROCESSING
from feature_extractor import RadiomicsFeatureExtractor
from habitat_analyzer import HabitatClusterAnalyzer


def load_roi_mask(roi_path):
    """
    Load and prepare ROI mask
    
    Parameters:
    -----------
    roi_path : str
        Path to ROI mask file
        
    Returns:
    --------
    tuple : (mask_array, mask_image)
    """
    if not os.path.exists(roi_path):
        raise FileNotFoundError(f"ROI file not found: {roi_path}")
    
    roi_img = sitk.ReadImage(roi_path)
    roi_arr = sitk.GetArrayFromImage(roi_img)
    roi_mask = roi_arr > 0  # Binary mask
    
    print(f"Loaded ROI with {roi_mask.sum()} voxels")
    return roi_mask, roi_img


def extract_features_parallel():
    """
    Batch feature extraction using parallel processing
    """
    # Define modality directories
    modality_dirs = {
        'T2': BASE_PATHS['T2'],
        'ADC': BASE_PATHS['ADC'],
        'DWI': BASE_PATHS['DWI']
    }
    mask_dir = BASE_PATHS['mask']
    output_dir = BASE_PATHS['output']
    
    # Get all mask files
    all_files = [f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')]
    
    if len(all_files) == 0:
        print("No mask files found!")
        return
    
    print(f"Found {len(all_files)} patients to process")
    
    # Prepare task arguments for parallel processing
    task_args = []
    for f in all_files:
        patient_id = f.replace('.nii.gz', '')
        task_args.append((f, modality_dirs, mask_dir, output_dir, patient_id))
    
    # Determine number of worker processes
    num_workers = PROCESSING['num_workers']
    if num_workers <= 0:
        num_workers = max(cpu_count() - 1, 1)
    
    print(f"Starting parallel processing with {num_workers} workers...")
    
    # Process in parallel with progress bar
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_process_single_patient, task_args),
            total=len(all_files),
            desc="Extracting features"
        ))
    
    # Print summary
    for res in results:
        print(res)


def _process_single_patient(args):
    """
    Helper function for parallel processing of single patient
    """
    filename, modality_dirs, mask_dir, output_dir, patient_id = args
    mask_path = os.path.join(mask_dir, filename)
    
    if not os.path.exists(mask_path):
        return f"[Skip] Mask not found: {filename}"
    
    # Initialize feature extractor
    extractor = RadiomicsFeatureExtractor()
    
    # Process each modality
    for modality, mod_dir in modality_dirs.items():
        image_path = os.path.join(mod_dir, filename)
        if not os.path.exists(image_path):
            print(f"[Skip] Image not found: {image_path}")
            continue
        
        # Extract and save features
        result = extractor.process_image_pair(
            image_path, mask_path, output_dir, patient_id, modality
        )
        print(result)
    
    return f"[Complete] {patient_id}"


def analyze_habitats_single_patient(patient_id, k_habitats=3):
    """
    Perform habitat analysis for a single patient
    
    Parameters:
    -----------
    patient_id : str
        Patient identifier
    k_habitats : int
        Number of habitats to identify
    """
    # Construct paths
    roi_path = os.path.join(BASE_PATHS['mask'], f"{patient_id}.nii.gz")
    patient_root = os.path.join(BASE_PATHS['output'], patient_id)
    
    # Load ROI mask
    roi_mask, _ = load_roi_mask(roi_path)
    
    # Initialize habitat analyzer
    analyzer = HabitatClusterAnalyzer(n_habitats=k_habitats)
    
    results = []
    
    # Process each modality
    for modality in PROCESSING['modalities']:
        mod_dir = os.path.join(patient_root, modality)
        
        if not os.path.isdir(mod_dir):
            print(f"[Skip] Modality directory not found: {mod_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Analyzing {modality} for patient {patient_id}")
        print(f"{'='*60}")
        
        # Output directory for habitat maps
        out_dir = os.path.join(patient_root, f"{modality}_habitats")
        
        # Perform habitat analysis
        try:
            metrics = analyzer.analyze_modality(mod_dir, roi_mask, out_dir)
            metrics["modality"] = modality
            metrics["patient"] = patient_id
            results.append(metrics)
            
            print(f"Completed {modality} habitat analysis")
            
        except Exception as e:
            print(f"Error processing {modality}: {str(e)}")
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        output_csv = os.path.join(patient_root, "habitat_metrics.csv")
        df.to_csv(output_csv, index=False)
        print(f"\nSaved metrics to: {output_csv}")
        
        # Display summary
        print("\nHabitat Analysis Summary:")
        print(df[['modality', 'habitat_1_ratio', 'habitat_2_ratio', 'habitat_3_ratio', 
                 'habitat_entropy_mean', 'habitat_edge_mean']].to_string())
    
    return results


def batch_habitat_analysis(patient_list=None, k_habitats=3):
    """
    Perform habitat analysis for multiple patients
    
    Parameters:
    -----------
    patient_list : list, optional
        List of patient IDs to process (if None, process all)
    k_habitats : int
        Number of habitats to identify
    """
    # If no patient list provided, find all patients with output
    if patient_list is None:
        output_dir = BASE_PATHS['output']
        patient_list = [d for d in os.listdir(output_dir) 
                       if os.path.isdir(os.path.join(output_dir, d))]
    
    all_results = []
    
    for patient_id in patient_list:
        print(f"\n{'#'*70}")
        print(f"Processing patient: {patient_id}")
        print(f"{'#'*70}")
        
        try:
            results = analyze_habitats_single_patient(patient_id, k_habitats)
            all_results.extend(results)
            
        except Exception as e:
            print(f"Failed to process {patient_id}: {str(e)}")
    
    # Save combined results
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_path = os.path.join(BASE_PATHS['output'], "all_patients_habitat_metrics.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"\nSaved combined results to: {combined_path}")
    
    return all_results


if __name__ == "__main__":
    """
    Main execution block
    """
    print("=" * 70)
    print("GLIOMA HABITAT ANALYSIS PIPELINE")
    print("=" * 70)
    
    # Example usage:
    # 1. Extract features for all patients
    # extract_features_parallel()
    
    # 2. Analyze habitats for specific patient
    analyze_habitats_single_patient("AnYuJu_FS_TRA", k_habitats=3)
    
    # 3. Batch analysis for multiple patients
    # patients = ["Patient1", "Patient2", "Patient3"]
    # batch_habitat_analysis(patients, k_habitats=3)