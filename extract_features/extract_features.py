from __future__ import division, print_function

import os
import sys

import numpy as np
import pandas as pd
import torch
import torchvision

from common import transforms as T
from common import utils
from eval_video_dataset import EvalVideoDataset
from models.model import Model

sys.path.insert(0, '..')


def extract_features(model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter=' ')
    header = 'Feature extraction:'
    with torch.no_grad():
        for sample in metric_logger.log_every(data_loader, 10, header, device=device):
            clip = sample['clip'].to(device, non_blocking=True)
            _, features = model(clip, return_features=True)
            data_loader.dataset.save_features(features, sample)


def main(args):
    print(args)
    print('TORCH VERSION: ', torch.__version__)
    print('TORCHVISION VERSION: ', torchvision.__version__)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print('LOADING DATA')
    transform = torchvision.transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((128, 171)),
        T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                    std=[0.22803, 0.22145, 0.216989]),
        T.CenterCrop((112, 112))
    ])
    metadata_df = pd.read_csv(args.metadata_csv_filename)
    shards = np.linspace(0, len(metadata_df), args.num_shards + 1).astype(int)
    start_idx, end_idx = shards[args.shard_id], shards[args.shard_id + 1]
    print(f'shard-id: {args.shard_id + 1} out of {args.num_shards}, '
          f'total number of videos: {len(metadata_df)}, shard size {end_idx - start_idx} videos')
    metadata_df = metadata_df.iloc[start_idx:end_idx].reset_index()
    metadata_df['is-computed-already'] = metadata_df['filename'].map(
        lambda f: os.path.exists(os.path.join(args.output_dir, os.path.basename(f).split('.')[0] + '.pkl')))
    metadata_df = metadata_df[metadata_df['is-computed-already'] is False].reset_index(drop=True)
    print(f'Number of videos to process after excluding the ones already computed on disk: {len(metadata_df)}')
    dataset = EvalVideoDataset(
        metadata_df=metadata_df,
        root_dir=args.data_path,
        clip_length=args.clip_length,
        frame_rate=args.frame_rate,
        stride=args.stride,
        output_dir=args.output_dir,
        transforms=transform)

    print('CREATING DATA LOADER')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    print(f'LOADING MODEL')
    if args.local_checkpoint:
        print(f'from the local checkpoint: {args.local_checkpoint}')
        pretrained_state_dict = torch.load(args.local_checkpoint, map_location='cpu')['model']
    else:
        raise Exception('No local checkpoint was set.')

    model = Model(backbone=args.backbone, num_classes=[1], num_heads=1, concat_gvf=False)
    model.to(device)
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'fc' not in k}
    state_dict = model.state_dict()
    pretrained_state_dict['fc.weight'] = state_dict['fc.weight']
    pretrained_state_dict['fc.bias'] = state_dict['fc.bias']
    model.load_state_dict(pretrained_state_dict)

    print('START FEATURE EXTRACTION')
    extract_features(model, data_loader, device)


if __name__ == '__main__':
    from opts import parse_args

    args = parse_args()
    main(args)
