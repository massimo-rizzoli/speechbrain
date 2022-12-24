"""
Downloads and creates data manifest files for Audio MNIST (spoken digit recognition).

Edit of the template in speechbrain/templates/speaker_id

Authors:
 * Mirco Ravanelli 2021
 * Massimo Rizzoli 2022
"""

import os
import json
import shutil
import logging
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio
from pathlib import Path
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

AUDIOMNIST_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/tags/v1.0.10.tar.gz"


def prepare_audioMNIST(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    split_ratio=[80, 10, 10],
):
    """
    Prepares the json files for the Mini Librispeech dataset.

    Downloads the dataset if it is not found in the `data_folder`.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the Mini Librispeech dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.

    Example
    -------
    >>> data_folder = '/path/to/AudioMNIST'
    >>> prepare_audioMNIST(data_folder, 'train.json', 'valid.json', 'test.json')
    """

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # If the dataset doesn't exist yet, download it
    folder = os.path.join(data_folder, "free-spoken-digit-dataset-1.0.10")
    if not check_folders(folder):
        download_audioMNIST(data_folder)

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    extension = [".wav"]
    wav_list = get_all_files(folder, match_and=extension)

    # Construct data dictionary
    data = get_data_dict(wav_list)

    # Random split the signal list into train, valid, and test sets.
    #data_split = split_sets(wav_list, split_ratio)
    data_split = split_sets(data, split_ratio)

    # Creating json files
    create_json(data_split["train"], save_json_train)
    create_json(data_split["valid"], save_json_valid)
    create_json(data_split["test"], save_json_test)


def get_data_dict(wav_list):
    """
    Creates a dictionary with each wav's file name as key.
    Each dictionary contains information about the entry.

    Arguments
    ---------
    wav_list: List[str]
        List of wav file paths in the dataset

    Returns
    ------
    dictionary containing a dictionary with information for each corresponding wav file
    """
    data = {}
    for wav in wav_list:
        f_name = Path(wav).stem
        [digit, spk, utt_num] = f_name.split('_')
        signal = read_audio(wav)
        data[f_name] = {
            "wav": wav,
            "digit": int(digit),
            "spk_id": spk,
            "utt_num": int(utt_num),
            "length": signal.shape[0]
        }
    return data

def get_keys_and_speakers(data):
    """
    Gets the lists of keys and speakers for the purpose of performing a stratified split

    Arguments
    ---------
    data: Dict[str, dict]
        Dictionary containing information for each wav in the dataset

    Returns
    ------
    keys: List[str]
        List containing the keys of each dataset entry
    speakers: List[str]
        List containing the speaker name of the corresponding entry
    """
    keys = []
    speakers = []
    for key, info in data.items():
        keys.append(key)
        speakers.append(info["spk_id"])
    return keys, speakers


def create_json(split, json_file):
    """
    Creates the json file given the data dictionary for one split.

    Arguments
    ---------
    split: Dict[str, dict]
        Data dictionary for one split
    json_file : str
        The path of the output json file
    """

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(split, json_f)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def split_sets(data, split_ratio):
    """
    Performs a stratified split of the shuffled data dictionary into training, validation, and test lists.

    Arguments
    ---------
    data: Dict[str, dict]
        Dictionary containing information for each wav in the dataset
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.

    Returns
    ------
    dictionary containing train, valid, and test splits.
    """
    # splits ratios
    [train_ratio, valid_ratio, test_ratio] = [ratio / 100 for ratio in split_ratio]
    rest_ratio = train_ratio + valid_ratio

    keys, speakers = get_keys_and_speakers(data)

    # stratified splits
    rest_keys, test_keys, rest_speakers, _ = train_test_split(keys, speakers, test_size=test_ratio, stratify=speakers)
    train_keys, valid_keys, _, _ = train_test_split(rest_keys, rest_speakers, test_size=valid_ratio/rest_ratio, stratify=rest_speakers)

    data_split = {}
    for split_name, split_keys in zip(["train", "valid", "test"], [train_keys, valid_keys, test_keys]):
        data_split[split_name] = {key:data[key] for key in split_keys}

    return data_split


def download_audioMNIST(destination):
    """Download dataset and unpack it.

    Arguments
    ---------
    destination : str
        Place to put the dataset.
    """
    archive = os.path.join(destination, "free-spoken-digit-dataset-1.0.10.tar.gz")
    download_file(AUDIOMNIST_URL, archive)
    shutil.unpack_archive(archive, destination)


if __name__ == "__main__":
    prepare_audioMNIST(".", "train.json", "valid.json", "test.json")
