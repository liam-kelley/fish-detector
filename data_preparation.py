import numpy as np
from tqdm import tqdm
import os
import argparse
from typing import List, Tuple

from util.path_util import deep_glob
from util.clean_util import clean_from_suffixes
from data_preprocessing.feature_extractor import Feature_Extractor


def download_DCASE24_Devolopment_Set(development_set_path : str):
    '''
    Downloads the DCASE24 Development set to the specified directory.

    Args:
        development_set_path (str): The directory path where the development set will be downloaded and unzipped.

    Notes:
        - The downloaded file is approximately 20+ GB in size.
    '''
    zipfile_path = os.path.join(development_set_path.split("/")[:-1], "Development_Set.zip")
    print(f"Downloading data to {zipfile_path}")
    print("Beware : 20+ Go")
    os.system(f"wget https://zenodo.org/record/6482837/files/Development_Set.zip?download=1 -O {zipfile_path}")
    os.system(f"unzip {zipfile_path}")
    print("Dataset is downloaded !")


def extract_features_from_wav_into_npy(wavfile_paths : List[str], features : List[str] = ["pcen"]):
    """
    Extract features from a list of WAV files and save them as Numpy arrays.

    This function processes each WAV file in the provided list, extracts various 
    audio features using a Feature_Extractor, and saves each feature to a separate 
    .npy file. The .npy files are saved in the same directory as the input WAV file 
    with the feature name appended to the original filename.

    Args:
        wavfile_paths (List[str]): A list of paths to the WAV files from which 
                                   features are to be extracted.

    Returns:
        None

    Side Effects:
        - Saves extracted features as .npy files in the same directory as the 
          original WAV files.
        - Prints the progress of feature extraction for each file.
        - Prints a list of files that encountered errors during processing.

    Exceptions:
        - If an error occurs during feature extraction, the error is printed, 
          the problematic file is added to the error list, and processing continues 
          with the next file.
        - If interrupted by a KeyboardInterrupt, the function exits immediately.

    Example:
        >>> extract_features_from_wav_into_npy(['path/to/file1.wav', 'path/to/file2.wav'])
        Processing file1.wav
        Processing file2.wav
        Encountered an error in these files:
        file_with_error.wav
    """
    
    print(f"Extracting features: {features}")
    
    # Inits
    fe = Feature_Extractor()
    error_files = []
    
    # For every file, try extracting and saving different features
    for file in (progress_bar := tqdm(wavfile_paths)):
        # Update progress_bar with name of file
        progress_bar.set_description(f"Processing {file.split("/")[-1]}") # pbar.set_description("Processing %s" % file.split("/")[-1])
        
        try: 
            extracted_features = fe.extract_features(file, features)
            
            for feature in extracted_features.keys():
                npy_path = file.replace(".wav", "_%s.npy" % feature)
                np.save(npy_path, extracted_features[feature])
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            # os.remove(file)
            print(e)
            error_files.append(file)
            breakpoint()
            continue
        
    print("Features extracted.")
    
    # In case of errors, give report
    if error_files:
        print("Encountered an error during extraction for these files:")
    for file in error_files:
        print(file.split("/")[-2] + "/" + file.split("/")[-1])
    

def calculate_feature_mean_std(array_list : List[np.ndarray]) -> Tuple[float, float]:
    """
    Calculate the mean and standard deviation of the elements in a list of numpy arrays.

    Args:
        array_list (List[np.ndarray]): A list of numpy arrays containing numerical data.

    Returns:
        Tuple[float, float]: A tuple containing the mean and standard deviation of all the elements in the provided arrays.

    Notes:
        - If the total number of elements in the array list is zero, the function ensures division by a small constant EPS to avoid division by zero.
        - EPS is assumed to be a predefined small constant (e.g., 1e-8) to prevent numerical instability.
    """
    EPS = 1e-8  # Small constant to prevent division by zero
    total_size = sum(array.size for array in array_list)
    N = max(float(total_size), EPS)

    total_sum = sum(array.sum() for array in array_list)
    mean_ = total_sum / N

    variance_sum = sum(((array - mean_) ** 2).sum() for array in array_list)
    std_ = np.sqrt(variance_sum / N)

    return mean_, std_


def normalize_features(wavfile_paths : List[str], features : List[str] = ["pcen"]):
    '''
    Normalize .npy features in-place.
    '''
    print(f"Normalizing features : {features}")
    for feature in features:
        print("Normalizing feature :", feature)
        array_list = []
        
        # Load all features into huge list
        for file in tqdm(wavfile_paths, desc="loading all npy into list"):
            try:
                npy_path = file.replace(".wav", "_%s.npy" % feature)
                array = np.load(npy_path)
                array_list.append(array)
            except:
                print("no such file", file)
                continue
        
        # Get mean and std
        mean, std = calculate_feature_mean_std(array_list)
        print("")
        del array_list
        
        # Normalize and save each file, replacing the original file
        for file in tqdm(wavfile_paths, desc="normalize and save files"):
            try:
                npy_path = file.replace(".wav", "_%s.npy" % feature)
                array = np.load(npy_path)
                array = (array - mean) / std
                np.save(npy_path, array)
            except:
                print("no such file", file)
                continue
    print("Features normalization is finished!")


def main(development_set_path: str):
    """
    Preprocesses DCASE development set wavs into useful features in npy format.
    """

    # Download dataset
    if not os.path.exists(development_set_path):
        download_DCASE24_Devolopment_Set(development_set_path)
    print("DCASE24 development set found.")
    
    # Clean unnecessary files
    clean_from_suffixes(path=development_set_path, suffixes=["Identifier"])
    
    # Extract features
    features = ["pcen"] # features = ["mel", "logmel", "pcen", "mfcc", "delta_mfcc"]
    wavfile_paths = deep_glob(development_set_path, ".wav")
    extract_features_from_wav_into_npy(wavfile_paths, features)
    normalize_features(wavfile_paths, features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--data-dir", default="./datasets/DCASE24/")
    args = parser.parse_args()
    
    development_set_path = os.path.join(args.data_dir, "Development_Set")
    main(development_set_path)