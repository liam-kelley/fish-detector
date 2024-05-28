import numpy as np
from tqdm import tqdm
import os
import argparse

from util.path_util import deep_glob
from data_preprocessing.feature_extractor import Feature_Extractor


EPS = 1e-8


def download_DCASE24_Devolopment_Set(development_set_path):
    zipfile_path = os.path.join(development_set_path.split("/")[:-1], "Development_Set.zip")
    print(f"Downloading data to {zipfile_path}")
    print("Beware : 20+ Go")
    os.system(f"wget https://zenodo.org/record/6482837/files/Development_Set.zip?download=1 -O {zipfile_path}")
    os.system(f"unzip {zipfile_path}")
    print("Dataset is downloaded !")


def extract_features_from_wav_into_npy(wavfile_paths : list):
    fe = Feature_Extractor()
    
    error_files = []
    for file in (pbar := tqdm(wavfile_paths)):
        pbar.set_description("Processing %s" % file.split("/")[-1])
        try:
            features = fe.extract_features(file)
            for feature in features.keys():
                npy_path = file.replace(".wav", "_%s.npy" % feature)
                np.save(npy_path, features[feature])
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            # os.remove(file)
            print(e)
            error_files.append(file)
            breakpoint()
            continue
        
    print("Encountered an error in these files:")
    for file in error_files:
        print(file.split("/")[-2] + "/" + file.split("/")[-1])


def calculate_feature_mean_std(array_list : list):
    assert(isinstance(array_list[0], np.ndarray))
    N = max(float(sum([array.size for array in array_list])), EPS)
    mean_ = sum([array.sum() for array in array_list]) / N
    std_ = np.sqrt(sum([((array - mean_) ** 2).sum() for array in array_list]) / N)
    return mean_, std_


def normalize_features(wavfile_paths : list, features : list):
    '''
    Normalize .npy features in-place.
    '''
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


def main(development_set_path: str):
    """Preprocesses DCASE development set wavs into useful features in npy format."""

    # Download dataset
    if not os.path.exists(development_set_path):
        download_DCASE24_Devolopment_Set(development_set_path)
    print("DCASE24 development set found.")
    
    # Extract features
    features = ["mel", "logmel", "pcen", "mfcc", "delta_mfcc"]
    print(f"Extracting features: {features}")
    wavfile_paths = deep_glob(development_set_path, ".wav")
    extract_features_from_wav_into_npy(wavfile_paths)
    print("Features extracted.")

    # Normalize features
    print(f"Normalizing features : {features}")
    normalize_features(wavfile_paths, features)
    print("Features normalization is finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--data-dir", default="./datasets/DCASE24/")
    args = parser.parse_args()
    
    development_set_path = os.path.join(args.data_dir, "Development_Set")
    main(development_set_path)