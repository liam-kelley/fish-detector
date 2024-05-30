import torch
from copy import deepcopy
from typing import Iterator
import random
import numpy as np

"""
Somewhat adapted from the Episodic batch sampler from https://github.com/jakesnell/prototypical-networks/
"""

class DCASE24_Task5_EpisodicBatchSampler(torch.utils.data.Sampler):
    def __init__(self, index_lists: dict[str, list[int]],
                        n_episodes: int = 2,
                        n_way: int = 5,
                        n_samples: int = 2,
                        oversample_smallest_classes: bool = True):
        """
        An episodic batch sampler appropriate for the dcase task 5 development set and dataset script.
        Can oversample underepresented classes, or undersample overrepresented classes.
        Automatically manages returning negative events if there is a presence of negative events in the index_lists.

        Args:
            index_lists (dict[str, list[int]]): List of indexes corresponding to each class,
                                                as can be returned by DCASE24_Task5_Dataset using
                                                the 'get_class_indexes_for_episodic_sampler' method.
                                                "CLS1_POS" : [idx1,idx2,...]
                                                "CLS1_NEG" : [idx1,idx2,...]
                                                "CLS2_POS" : [idx1,idx2,...]
                                                ...
            n_episodes (int): Number of episodes present in a training batch to load
            n_way (int): Number of classes in a episode.
            n_samples (int): Number of support + query samples per class in a episode (For meta-training and meta-validation, both during training)
            oversample_smallest_classes (bool): Strategy regarding oversampling.
            return_negative_events (bool): Strategy regarding negative events.
        """
        self.index_lists = index_lists
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_samples = n_samples
        self.oversample_smallest_classes = oversample_smallest_classes        
        
        # Get class names
        class_names_with_pos_neg = list(self.index_lists.keys())
        self.class_names = [name[:-4] for name in class_names_with_pos_neg]
        # Identify if returning negative events is needed or not. (TODO: maybe do this more safely)
        pos_neg_set = set([name[-3:] for name in class_names_with_pos_neg])
        if "NEG" in pos_neg_set : self.return_negative_events = True
        else: self.return_negative_events = False
        
        # Calculate episode size and batch size
        self.episode_size = self.n_way * self.n_samples
        if self.return_negative_events : self.episode_size *= 2
        self.batch_size = self.n_episodes * self.episode_size
        
        # Init
        self.temp_index_lists = {}
        
        # Manage oversampling classes or undersampling classes
        self.n_events_per_class = {cls : len(index_list) for cls, index_list in self.index_lists.items()}
        self.max_events = max(self.n_events_per_class.values())
        self.min_events = min(self.n_events_per_class.values())
             
    def _set_temp_index_lists_before_every_epoch(self):
        '''
        Reset indexes each epoch
        Manage oversampling / undersampling (repeating / cropping)
        Shuffling
        '''
        # Reset index lists
        self.temp_index_lists = deepcopy(self.index_lists)
        
        # Manage oversampling classes or undersampling classes
        if self.oversample_smallest_classes:
            # Repeat smallest index lists until as long as max_events
            for cls, index_list in self.temp_index_lists.items():
                repeat_count = (self.max_events // self.n_events_per_class[cls]) + 1
                extended_list = index_list * repeat_count
                
                random_start_point = random.randint(0, self.n_events_per_class[cls] - 1)
                self.temp_index_lists[cls] = extended_list[random_start_point : random_start_point + self.max_events]
        else:
            # Crop biggest index lists until as long as min_events
            for cls, index_list in self.temp_index_lists.items():
                random_start_point = random.randint(0, self.min_events - 1)
                self.temp_index_lists[cls] = index_list[random_start_point : random_start_point + self.min_events]  
                
        # Shuffle index lists
        for cls in self.temp_index_lists:
            random.shuffle(self.temp_index_lists[cls])       
        
    def __len__(self):
        # Manage oversampling changing the amount of effective events
        if self.oversample_smallest_classes:
            len_events = self.max_events
        else:
            len_events = self.min_events
        
        # Manage how effective batch_size changes when returning negative events.
        positive_batch_size = self.batch_size
        if self.return_negative_events:
            positive_batch_size = self.batch_size // 2
        
        return len_events // positive_batch_size # Last batch is dropped

    def __iter__(self) -> Iterator[list[int]]:
        '''
        Directly yields full batches of indexes for dataloader.
        Batch format : flattened list of lists of lists. shape(n_episodes, n_way (POS then NEG if needed), n_samples)
        
        You can get the relevant class name from the idx along with the item from
            the __getitem__ method of the dataset object.
        '''
        self._set_temp_index_lists_before_every_epoch()
        for batch_idx in range(self.__len__()):
            batch = []
            for episode_idx in range(self.n_episodes):
                episode = []
                
                # Choose n_way classes per episode
                n_way_class_names = random.sample(self.class_names, self.n_way)
                
                # Figure out which samples to get (samples are pre-shuffled every epoch)
                event_slice_start = batch_idx * self.batch_size + episode_idx * self.episode_size
                event_slice_end = event_slice_start + self.n_samples
                
                for cls in n_way_class_names:
                    # Get n_samples positive samples of a class chosen for this episode
                    episode.append(self.temp_index_lists[f"{cls}_POS"][event_slice_start : event_slice_end])
                    
                    if self.return_negative_events:
                        # Get n_samples negative samples of a class chosen for this episode
                        episode.append(self.temp_index_lists[f"{cls}_NEG"][event_slice_start : event_slice_end])
                
                batch.append(episode)
                
            # flatten list of lists of lists
            batch = np.array(batch).flatten().tolist()
            yield batch
            
        # Last batch is dropped