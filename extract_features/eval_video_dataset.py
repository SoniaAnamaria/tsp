from __future__ import division, print_function

import os
import pickle as pkl

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video


class EvalVideoDataset(Dataset):
    def __init__(self, metadata_df, root_dir, clip_length, frame_rate, stride, output_dir, transforms=None):
        metadata_df = EvalVideoDataset._append_root_dir_to_filenames_and_check_files_exist(metadata_df, root_dir)
        self.clip_metadata_df = EvalVideoDataset._generate_clips_metadata(metadata_df, clip_length, frame_rate, stride)
        self.clip_length = clip_length
        self.frame_rate = frame_rate
        self.stride = stride
        self.output_dir = output_dir
        self.transforms = transforms
        self.saved_features = {}

    def __len__(self):
        return len(self.clip_metadata_df)

    def __getitem__(self, idx):
        sample = {}
        row = self.clip_metadata_df.iloc[idx]
        filename, fps, clip_start, is_last_clip = row['filename'], row['fps'], row['clip-t-start'], row['is-last-clip']
        clip_end = clip_start + self.clip_length / self.frame_rate

        frames, _, _ = read_video(filename=filename, start_pts=clip_start, end_pts=clip_end, pts_unit='sec')
        idx = EvalVideoDataset._resample_video_idx(self.clip_length, fps, self.frame_rate)
        frames = frames[idx][:self.clip_length]
        if frames.shape[0] != self.clip_length:
            raise RuntimeError(f'<EvalVideoDataset>: got clip of length {frames.shape[0]} != {self.clip_length}.'
                               f'filename={filename}, clip_t_start={clip_start}, clip_t_end={clip_end}, fps={fps} ')

        sample['clip'] = self.transforms(frames)
        sample['filename'] = filename
        sample['is-last-clip'] = is_last_clip
        return sample

    def save_features(self, batch_features, batch_input):
        batch_features = batch_features.detach().cpu().numpy()

        for i in range(batch_features.shape[0]):
            filename, is_last_clip = batch_input['filename'][i], batch_input['is-last-clip'][i]
            if not (filename in self.saved_features):
                self.saved_features[filename] = []
            self.saved_features[filename].append(batch_features[i, ...])

            if is_last_clip:
                output_filename = os.path.join(self.output_dir, os.path.basename(filename).split('.')[0] + '.pkl')
                self.saved_features[filename] = np.stack(self.saved_features[filename])
                with open(output_filename, 'wb') as f:
                    pkl.dump(self.saved_features[filename], f)
                del self.saved_features[filename]

    @staticmethod
    def _append_root_dir_to_filenames_and_check_files_exist(df, root_dir):
        df['filename'] = df['filename'].map(lambda f: os.path.join(root_dir, f))
        filenames = df.drop_duplicates('filename')['filename'].values
        for f in filenames:
            if not os.path.exists(f):
                raise ValueError(f'<EvalVideoDataset>: file={f} does not exists. '
                                 f'Double-check root_dir and metadata_df inputs')
        return df

    @staticmethod
    def _generate_clips_metadata(df, clip_length, frame_rate, stride):
        clip_metadata = {
            'filename': [],
            'fps': [],
            'clip-t-start': [],
            'is-last-clip': [],
        }
        for i, row in df.iterrows():
            total_frames_after_resampling = int(row['video-frames'] * (float(frame_rate) / row['fps']))
            idx = EvalVideoDataset._resample_video_idx(total_frames_after_resampling, row['fps'], frame_rate)
            if isinstance(idx, slice):
                frame_idx = np.arange(row['video-frames'])[idx]
            else:
                frame_idx = idx.numpy()
            clip_t_start = list(frame_idx[np.arange(0, frame_idx.shape[0] - clip_length + 1, stride)] / row['fps'])
            num_clips = len(clip_t_start)

            clip_metadata['filename'].extend([row['filename']] * num_clips)
            clip_metadata['fps'].extend([row['fps']] * num_clips)
            clip_metadata['clip-t-start'].extend(clip_t_start)
            is_last_clip = [0] * (num_clips - 1) + [1]
            clip_metadata['is-last-clip'].extend(is_last_clip)

        return pd.DataFrame(clip_metadata)

    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            step = int(step)
            return slice(None, None, step)
        idx = torch.arange(num_frames, dtype=torch.float32) * step
        return idx.floor().to(torch.int64)
