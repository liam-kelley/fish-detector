import torch
import os
import glob
from util.path_util import deep_glob
import tqdm
import pandas as pd
from itertools import chain
import librosa
import numpy as np

class DCASE24_Task5_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 paths: dict = {"train_dir": "datasets/DCASE24/Development_Set/Training_Set",
                                "val_dir": "datasets/DCASE24/Development_Set/Validation_Set",},
                 features: list[str] = ["pcen"],
                 train_param: dict = {"n_shot": 5,
                                    "k_way": 1,
                                    "segment_len": 100},
                 dataset_param: dict = {"margin_around_sequences": 0.25,
                                        "remove_short_max_negative_duration": True,
                                        "remove_short_max_negative_duration_threshold": 0.3,
                                        "mode": None},
                 preprocessing_param: dict = {"pcen_frames_per_second" : 50}
                 ):
        """
        Useful for all your task5 needs.
        TODO : loading multiple types of features at once. But right now I'm mostly interested in pcen.
        """
        self.paths = paths
        self.features = features
        self.train_param = train_param
        self.dataset_param = dataset_param
        self.preprocessing_param = preprocessing_param
        
        ##### GET CSV FILES #####

        self.train_csv_files = deep_glob(self.paths["train_dir"], suffix=".csv")
        self.val_csv_files = deep_glob(self.paths["val_dir"], suffix=".csv")
        self.all_csv_files = self.train_csv_files + self.val_csv_files
        
        ##### BUILD META DATA #####
        
        self.train_classes = []
        self.val_classes = []
        self.meta = {} # Metadata dictionnary. See structure in self._build_meta docstring.
        self._build_meta()
        self.all_classes = list(self.meta.keys())
        
        # Create some maps to have numerical values in the dataframe instead of strings
        self.POS_NEG_map = {"NEG": 0, "POS": 1}
        self.class_name_map = {label: idx for idx, label in enumerate(self.all_classes)} # key = class name, value = unique int
        self.class_index_map = {v: k for k, v in self.class_name_map.items()} # key = unique int, value = class name

        self.df = self._convert_meta_dict_to_event_dataframe()
        del self.meta # df is preferred
        
        ##### SET A SPECIFIC MODE (train, val) #####
        
        self.archival_df = None
        if self.dataset_param["mode"] == "train":
            self.set_training_mode()
        elif self.dataset_param["mode"] == "val":
            self.set_validation_mode()
    
    def _get_df_positive_rows(self, file : str) -> pd.DataFrame:
        """
        Reads a CSV file and filters rows containing any 'POS' value.

        Parameters:
        file (str): The path to the CSV file.

        Returns:
        pandas.DataFrame: A DataFrame containing only the rows where any value is 'POS'.
        """
        df = pd.read_csv(file, header=0, index_col=False)
        return df[(df == "POS").any(axis=1)]
    
    def _get_times(self, df : pd.DataFrame) -> tuple[list[float],list[float]]:
        """
        Adds a Margin of 25 ms around the events onsets and offsets of event
        dataframe, and returns them as a list.
        
        Parameters:
        df (pandas.DataFrame): DataFrame with "Starttime" and "Endtime" columns.

        Returns:
        tuple: Two lists with adjusted start and end times.
        """
        df["Starttime"] = df["Starttime"] - self.dataset_param["margin_around_sequences"]
        df["Endtime"] = df["Endtime"] + self.dataset_param["margin_around_sequences"]
        start_time = [value for value in df["Starttime"]]
        end_time_list = [value for value in df["Endtime"]]
        return start_time, end_time_list
    
    def _create_class_list(df_positives: pd.DataFrame, sub_dataset_name: str, start_time_list: list[float]) -> list[str]:
        """
        Creates a list of classes based on the columns that have positives in the DataFrame.
        
        Parameters:
        - df_positives (pd.DataFrame): DataFrame containing positive samples with class labels.
        - sub_dataset_name (str): Name of the sub-dataset used to label the classes for validation.
        - start_time_list (list): List of start times corresponding to the data samples.
        
        Returns:
        - cls_list (list): List of class labels
        """ 
        #  To differentiate classes in the Validation Dataset, because all classes are categorized as "Q", so use subdataset name instead (HB, ME...)
        if df_positives.columns[3] == "Q":  
            cls_list = [sub_dataset_name] * len(start_time_list)
        else:
            # Use apply to get the columns where value is "POS"
            cls_list = df_positives.apply(lambda row: row[row == "POS"].index.tolist(), axis=1)
            # Flatten Series into a list
            cls_list = cls_list.explode().tolist()
    
        return cls_list

    def _update_meta(self, start_time_list: list[float], end_time_list: list[float], cls_list: list[str], csv_file: str) -> None:
        """
        Provided a list of start times, end times and classes for different events,
        for every event, add positive and negative metadata to the metadata dictionnary.

        Args:
            start_time_list (list[float]): _description_
            end_time_list (list[float]): _description_
            cls_list (list[str]): _description_
            csv_file (str): _description_
        """
        
        audio_path = csv_file.replace("csv", "wav")

        # Re-init all current negative start times for correct negative event computation
        for cls in set(cls_list):
            if cls in self.meta.keys():
                self.meta[cls]["neg_start_time"] = 0
        
        # Initialize new classes if needed
        for cls in cls_list:
            # add classes for (train, val and extra train) to appropriate list of classes
            if (csv_file in self.train_csv_files) and (cls not in self.train_classes):
                self.train_classes.append(cls)
            if (csv_file in self.val_csv_files) and (cls not in self.val_classes):
                self.val_classes.append(cls)

            # If the class isn't yet in the meta data, initialize all meta stuff for it
            if cls not in self.meta.keys():
                self.meta[cls] = {}
                self.meta[cls]["temp_neg_start_time"] = 0  # temp variable
                self.meta[cls]["pos_info"] = []  # positive segment onset and offset
                self.meta[cls]["neg_info"] = []  # negative segment onset and offset
                self.meta[cls]["pos_duration"] = []  # duration of positive segments
                self.meta[cls]["neg_duration"] = []  # duration of negative segments
                self.meta[cls]["total_audio_duration"] = []  # duration
                self.meta[cls]["pos_file"] = []  # filename
                self.meta[cls]["neg_file"] = []  # filename

        # For every event, add positive AND negative metadata.
        for start, end, cls in zip(start_time_list, end_time_list, cls_list):
            self.meta[cls]["total_audio_duration"].append(librosa.get_duration(filename=audio_path, sr=None))
            self.meta[cls]["info"].append((start, end))
            self.meta[cls]["duration"].append(end - start)
            neg_start = np.clip(
                self.meta[cls]["temp_neg_start_time"] - 0.025,
                a_min=0,
                a_max=None)
            neg_end = np.clip(
                start + 0.025,
                a_min=None,
                a_max=self.meta[cls]["total_audio_duration"][-1])
            self.meta[cls]["neg_info"].append((neg_start, neg_end))
            self.meta[cls]["neg_duration"].append(neg_end - neg_start)
            self.meta[cls]["pos_file"].append(audio_path) # filename
            self.meta[cls]["neg_file"].append(audio_path) # filename
            self.meta[cls]["temp_neg_start_time"] = end # next neg start time will be the end of this positive segment.

        # Add thess lines if the data in the validation set is sparse
        # for cls in cls_list:
        #     if(np.sum(self.meta[cls]["neg_duration"]) < 2.0):
        #         print("The annotated negative sample of %s is less then 2.0 seconds, use all remaining part as negative training set" % audio_path)
        #         neg_start, neg_end = np.clip(self.meta[cls]["temp_neg_start_time"]-0.025, a_min=0, a_max=None), self.meta[cls]["total_audio_duration"][-1]
        #         self.meta[cls]["neg_info"].append((neg_start, neg_end))
        #         self.meta[cls]["neg_duration"].append(neg_end - neg_start)
        #         self.meta[cls]["neg_file"].append(audio_path)

        # for cls in cls_list:
        #     # Add thess lines if the data in the validation set is sparse
        #     if(np.sum(self.meta[cls]["neg_duration"]) < 30.0):
        #         print("The annotated negative sample of %s is less then 30.0 seconds, use all remaining part as negative training set" % audio_path)
        #         # neg_start, neg_end = np.clip(self.meta[cls]["temp_neg_start_time"]-0.025, a_min=0, a_max=None), self.meta[cls]["total_audio_duration"][-1]
        #         # self.meta[cls]["neg_info"].append((neg_start, neg_end))
        #         # self.meta[cls]["neg_duration"].append(neg_end - neg_start)
        #         # self.meta[cls]["info"].append(self.meta[cls]["info"][-1])
        #         # self.meta[cls]["duration"].append(self.meta[cls]["duration"][-1])
        #         # self.meta[cls]["total_audio_duration"].append(librosa.get_duration(filename=audio_path,sr=None))
        #         # self.meta[cls]["pos_file"].append(audio_path)
        #         self.build_negative_based_on_energy(self.fe.extract_feature(audio_path,"logmel"), cls, audio_path)

    def _remove_short_max_negative_duration(self):
        '''
        Remove classes with short negative durations
        '''
        delete_keys = []
        for cls in self.meta.keys():
            if max_neg_duration := max(self.meta[cls]["neg_duration"]) < self.dataset_param["remove_short_max_negative_duration_threshold"]:
                delete_keys.append(cls)
                
        for cls in delete_keys:
            del self.meta[cls]
            print(f"Deleted class {cls} due to short max negative length. (max < {self.dataset_param["remove_short_max_negative_duration_threshold"]}")

    def _build_meta(self):
        """
        Builds the metadata dictionnary in self.meta, and builds self.train_classes and self.val_classes lists.
        
        meta dictionnary structure
        {
            <class-name>: {
                "pos_info" # positive segment onset and offset : [(<start-time>, <end-time>), ...]
                "neg_info" # negative segment onset and offset : [(<start-time>, <end-time>), ...]
                "pos_duration" # duration of positive segments : [duration1, duration2, ...]
                "neg_duration" # duration of negative segments : [duration1, duration2, ...]
                "total_audio_duration" # duration
                "pos_file" # [filename1, filename2, ...]
                "neg_file" # [filename1, filename2, ...]
            }
        }
        """
        print("Building meta data...")
        
        for file in tqdm(self.all_csv_files):
            df_positives = self._get_df_positive_rows(file)
            start_time_list, end_time_list = self._get_times(df_positives)
            
            sub_dataset_name = file.split("/")[-2]
            cls_list = self._create_class_list(df_positives, sub_dataset_name, start_time_list)
            self._update_meta(start_time_list, end_time_list, cls_list, file)
    
        if self.dataset_param["remove_short_max_negative_duration"]:
            self._remove_short_max_negative_duration()
        
        print("Meta data build done.")
        
    def _convert_meta_dict_to_event_dataframe(self):
        event_dict = {
            "class": [],
            "POS_NEG": [],
            "filename": [],
            "start_time": [],
            "end_time": []
        }

        for class_name, data in self.meta.items():
            for (start_time, end_time), pos_file in zip(data["pos_info"], data["pos_file"]):
                event_dict["class"].append(self.class_name_map[class_name])
                event_dict["POS_NEG"].append(self.POS_NEG_map["POS"])
                event_dict["filename"].append(pos_file)
                event_dict["start_time"].append(start_time)
                event_dict["end_time"].append(end_time)

            for (start_time, end_time), neg_file in zip(data["neg_info"], data["neg_file"]):
                event_dict["class"].append(self.class_name_map[class_name])
                event_dict["POS_NEG"].append(self.POS_NEG_map["NEG"])
                event_dict["filename"].append(neg_file)
                event_dict["start_time"].append(start_time)
                event_dict["end_time"].append(end_time)

        print("Converted meta dict to an event Dataframe")

        return pd.DataFrame(event_dict)

    def _get_pcen_segment(self, event_info):
        # init event info
        pcen_filename = event_info['filename'].split(".")[-2] + "_pcen.npy"
        start_frame = int(event_info["start_time"] * self.preprocessing_param["pcen_frames_per_second"])
        if start_frame < 0 :
            start_frame = 0
        end_frame = int(event_info["end_time"] * self.preprocessing_param["pcen_frames_per_second"])
        event_duration = end_frame - start_frame
        seg_len = self.train_param["segment_len"]
        
        # Prepare loading from memory.
        pcen_memmap = np.load(pcen_filename, mmap_mode='r')
        
        # If too small duration, tile; if too big, take random segment insde the event
        if event_duration < seg_len:
            # Load pcen
            pcen = np.array(pcen_memmap[start_frame:end_frame])
            # Tile until big enough
            tile_times = np.ceil(seg_len / event_duration)
            pcen = np.tile(pcen, (int(tile_times), 1)) # tile along time axis
            # Crop to correct length
            pcen = pcen[:seg_len]
        else:
            # get random start frame
            rand_start_frame = np.random.randint(low=start_frame, high=end_frame - seg_len)
            # Load pcen
            pcen = pcen_memmap[rand_start_frame : rand_start_frame + seg_len]
        
        # sanity check
        assert pcen.shape[0] == seg_len
        return pcen

    def __len__(self):
        '''Returns total number of positive events.'''
        return len(self.df[self.df["POS_NEG"] == self.POS_NEG_map["POS"]])
    
    def __getitem__(self, idx):
        # Load an event
        event_info = self.df.iloc[idx] # "class", "POS_NEG", "filename" ,"start_time" ,"end_time"
        
        # Get segment from precomputed pcen filename beginning at start_time and ending at end time (fixed segment length)
        pcen_segment = self._get_pcen_segment(event_info)
    
        # Get class name + positivity or negativity of segment
        event_class_name = self.class_index_map[event_info["class"]]
        event_pos_neg = event_info["POS_NEG"]
        
        return pcen_segment, event_class_name, event_pos_neg

    def set_training_mode(self):
        '''
        Updates the dataframe to only load training classes.
        '''
        if not self.archival_df:
            self.archival_df = self.df.copy()
            
        train_classes_numerical = [self.class_name_map[cls] for cls in self.train_classes]
        
        # Only keep training classes
        self.df = self.archival_df[self.archival_df["class"] in train_classes_numerical]
    
    def set_validation_mode(self):
        '''
        Updates the dataframe to only load validation classes.
        '''
        if not self.archival_df:
            self.archival_df = self.df.copy()
            
        val_classes_numerical = [self.class_name_map[cls] for cls in self.val_classes]
        
        # Only keep validation classes
        self.df = self.archival_df[self.archival_df["class"] in val_classes_numerical]
        
    def set_full_default_mode(self):
        '''
        Updates the dataframe to load all classes. Why would you need this? Idk
        Only useful if you previously transformed this dataset into a training or validation mode.
        '''
        if self.archival_df:
            self.df = self.archival_df.copy()

    def get_class_indexes_for_sampler(self):
        raise NotImplementedError
