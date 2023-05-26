from __future__ import division, print_function

import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video


class UntrimmedVideoDataset(Dataset):
    def __init__(self, csv_filename, root_dir, clip_length, frame_rate, clips_per_segment, temporal_jittering,
                 label_columns, label_mappings, seed=42, transforms=None, global_video_features=None, debug=False):
        df = UntrimmedVideoDataset._clean_df_and_remove_short_segments(pd.read_csv(csv_filename), clip_length,
                                                                       frame_rate)
        self.df = UntrimmedVideoDataset._append_root_dir_to_filenames_and_check_files_exist(df, root_dir)
        self.clip_length = clip_length
        self.frame_rate = frame_rate
        self.clips_per_segment = clips_per_segment
        self.temporal_jittering = temporal_jittering
        self.rng = np.random.RandomState(seed=seed)
        self.uniform_sampling = np.linspace(0, 1, clips_per_segment)
        self.transforms = transforms
        self.label_columns = label_columns
        self.label_mappings = label_mappings
        for label_column, label_mapping in zip(label_columns, label_mappings):
            self.df[label_column] = self.df[label_column].map(lambda x: -1 if pd.isnull(x) else label_mapping[x])
        self.global_video_features = global_video_features
        self.debug = debug

    def __len__(self):
        if self.debug:
            return 100
        return len(self.df) * self.clips_per_segment

    def __getitem__(self, idx):
        sample = {}
        row = self.df.iloc[idx % len(self.df)]
        filename, fps, t_start, t_end = row['filename'], row['fps'], row['t-start'], row['t-end']
        if self.temporal_jittering:
            ratio = self.rng.uniform()
        else:
            ratio = self.uniform_sampling[idx // len(self.df)]
        clip_t_start = t_start + ratio * (t_end - t_start - self.clip_length / self.frame_rate)
        clip_t_end = clip_t_start +  self.clip_length / self.frame_rate

        frames, _, _ = read_video(filename=filename, start_pts=clip_t_start, end_pts=clip_t_end, pts_unit='sec')
        idx = UntrimmedVideoDataset._resample_video_idx(self.clip_length, fps, self.frame_rate)
        frames = frames[idx][:self.clip_length]
        if frames.shape[0] != self.clip_length:
            raise RuntimeError(f'<UntrimmedVideoDataset>: got clip of length {frames.shape[0]} != {self.clip_length}.'
                               f'filename={filename}, clip_t_start={clip_t_start}, clip_t_end={clip_t_end}, fps={fps} '
                               f't_start={t_start}, t_end={t_end}')

        sample['clip'] = self.transforms(frames)
        for label_column in self.label_columns:
            sample[label_column] = row[label_column]
        if self.global_video_features:
            f = h5py.File(self.global_video_features, 'r')
            sample['gvf'] = torch.tensor(f[os.path.basename(filename).split('.')[0]][()])
            f.close()
        return sample

    @staticmethod
    def _clean_df_and_remove_short_segments(df, clip_length, frame_rate):
        df['t-start'] = np.maximum(df['t-start'], 0)
        df['t-end'] = np.minimum(df['t-end'], df['video-duration'])
        segment_length = frame_rate * (df['t-end'] - df['t-start'])
        mask = segment_length >= clip_length
        num_segments = len(df)
        num_segments_to_keep = sum(mask)
        if num_segments - num_segments_to_keep > 0:
            df = df[mask].reset_index(drop=True)
            print(f'<UntrimmedVideoDataset>: removed {num_segments - num_segments_to_keep}='
                  f'{100 * (1 - num_segments_to_keep / num_segments):.2f}% from the {num_segments} '
                  f'segments from the input CSV file because they are shorter than '
                  f'clip_length={clip_length} frames using frame_rate={frame_rate} fps.')
        return df

    @staticmethod
    def _append_root_dir_to_filenames_and_check_files_exist(df, root_dir):
        df['filename'] = df['filename'].map(lambda f: os.path.join(root_dir, f))
        filenames = df.drop_duplicates('filename')['filename'].values
        for f in filenames:
            if not os.path.exists(f):
                raise ValueError(f'<UntrimmedVideoDataset>: file={f} does not exists. '
                                 f'Double-check root_dir and csv_filename inputs.')
        return df

    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            step = int(step)
            return slice(None, None, step)
        idx = torch.arange(num_frames, dtype=torch.float32) * step
        return idx.floor().to(torch.int64)
