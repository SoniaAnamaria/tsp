from __future__ import division, print_function

import datetime
import json
import os
import sys
import time
from itertools import chain

import numpy as np
import torch
import torchvision

from common import transforms as T
from common import utils
from common.scheduler import WarmupMultiStepLR
from models.model import Model
from untrimmed_video_dataset import UntrimmedVideoDataset

sys.path.insert(0, '..')


def compute_accuracies_and_log_metrics(metric_logger, loss, outputs, targets, head_losses, label_columns):
    for output, target, head_loss, label_column in zip(outputs, targets, head_losses, label_columns):
        mask = target != -1
        output, target = output[mask], target[mask]
        if output.shape[0]:
            head_acc, = utils.accuracy(output, target, top_k=(1,))
            metric_logger.meters[f'acc_{label_column}'].update(head_acc.item(), n=output.shape[0])
        metric_logger.meters[f'loss_{label_column}'].update(head_loss.item())
    metric_logger.update(loss=loss.item())


def write_metrics_results_to_file(metric_logger, epoch, label_columns, output_dir):
    results = f'** Valid Epoch {epoch}: '
    accuracies = []
    for label_column in label_columns:
        results += f' <{label_column}> Accuracy {metric_logger.meters[f"acc_{label_column}"].global_avg:.3f}'
        results += f' Loss {metric_logger.meters[f"loss_{label_column}"].global_avg:.3f};'
        accuracies.append(metric_logger.meters[f'acc_{label_column}'].global_avg)
    results += f' Total Loss {metric_logger.meters["loss"].global_avg:.3f}'
    avg_acc = np.average(accuracies)
    results += f' Avg Accuracy {avg_acc:.3f}\n'
    utils.write_to_file(file=os.path.join(output_dir, 'results.txt'), mode='a', content_to_write=results)
    return results


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq, label_columns,
                    loss_alphas):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=' ')
    for g in optimizer.param_groups:
        metric_logger.add_meter(f'{g["name"]}_lr', utils.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))
    header = f'Train Epoch {epoch}:'
    for sample in metric_logger.log_every(data_loader, print_freq, header, device=device):
        start_time = time.time()

        clip = sample['clip'].to(device)
        if 'gvf' in sample:
            gvf = sample['gvf'].to(device)
        else:
            gvf = None
        targets = []
        for x in label_columns:
            targets.append(sample[x].to(device))
        outputs = model(clip, gvf=gvf)

        head_losses, loss = [], 0
        for output, target, alpha in zip(outputs, targets, loss_alphas):
            head_loss = criterion(output, target)
            head_losses.append(head_loss)
            loss += alpha * head_loss

        for param in model.parameters():
            param.grad = None
        loss.backward()
        optimizer.step()

        compute_accuracies_and_log_metrics(metric_logger, loss, outputs, targets, head_losses, label_columns)
        for g in optimizer.param_groups:
            metric_logger.meters[f'{g["name"]}_lr'].update(g['lr'])
        metric_logger.meters['clips/s'].update(clip.shape[0] / (time.time() - start_time))
        lr_scheduler.step()


def evaluate(model, criterion, data_loader, device, epoch, print_freq, label_columns, loss_alphas, output_dir):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter=' ')
    header = f'Valid Epoch {epoch}:'
    with torch.no_grad():
        for sample in metric_logger.log_every(data_loader, print_freq, header, device=device):
            clip = sample['clip'].to(device, non_blocking=True)
            if 'gvf' in sample:
                gvf = sample['gvf'].to(device, non_blocking=True)
            else:
                gvf = None
            targets = []
            for x in label_columns:
                targets.append(sample[x].to(device, non_blocking=True))
            outputs = model(clip, gvf=gvf)

            head_losses, loss = [], 0
            for output, target, alpha in zip(outputs, targets, loss_alphas):
                head_loss = criterion(output, target)
                head_losses.append(head_loss)
                loss += alpha * head_loss

            compute_accuracies_and_log_metrics(metric_logger, loss, outputs, targets, head_losses, label_columns)
    results = write_metrics_results_to_file(metric_logger, epoch, label_columns, output_dir)
    print(results)


def main(args):
    print(args)
    print('TORCH VERSION: ', torch.__version__)
    print('TORCHVISION VERSION: ', torchvision.__version__)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    train_dir = os.path.join(args.root_dir, args.train_subdir)
    valid_dir = os.path.join(args.root_dir, args.valid_subdir)

    print('LOADING DATA')
    label_mappings = []
    for label_mapping_json in args.label_mapping_jsons:
        with open(label_mapping_json) as f:
            label_mapping = json.load(f)
            label_mappings.append(dict(zip(label_mapping, range(len(label_mapping)))))
    if args.backbone == 'r2plus1d_34':
        transform_train = torchvision.transforms.Compose([
            T.ToFloatTensorInZeroOne(),
            T.Resize((128, 171)),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989]),
            T.RandomCrop((112, 112))
        ])
    else:
        transform_train = torchvision.transforms.Compose([
            T.ToFloatTensorInZeroOne(),
            T.Resize((256, 342)),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989]),
            T.RandomCrop((224, 224))
        ])
    dataset_train = UntrimmedVideoDataset(
        csv_filename=args.train_csv_filename,
        root_dir=train_dir,
        clip_length=args.clip_len,
        frame_rate=args.frame_rate,
        clips_per_segment=args.clips_per_segment,
        temporal_jittering=True,
        transforms=transform_train,
        label_columns=args.label_columns,
        label_mappings=label_mappings,
        global_video_features=args.global_video_features)
    if args.backbone == 'r2plus1d_34':
        transform_valid = torchvision.transforms.Compose([
            T.ToFloatTensorInZeroOne(),
            T.Resize((128, 171)),
            T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989]),
            T.CenterCrop((112, 112))
        ])
    else:
        transform_valid = torchvision.transforms.Compose([
            T.ToFloatTensorInZeroOne(),
            T.Resize((256, 342)),
            T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989]),
            T.CenterCrop((224, 224))
        ])
    dataset_valid = UntrimmedVideoDataset(
        csv_filename=args.valid_csv_filename,
        root_dir=valid_dir,
        clip_length=args.clip_len,
        frame_rate=args.frame_rate,
        clips_per_segment=args.clips_per_segment,
        temporal_jittering=False,
        transforms=transform_valid,
        label_columns=args.label_columns,
        label_mappings=label_mappings,
        global_video_features=args.global_video_features)

    print('CREATING DATA LOADERS')
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                    sampler=None, num_workers=args.workers, pin_memory=True)

    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False,
                                                    sampler=None, num_workers=args.workers, pin_memory=True)

    print('CREATING MODEL')
    cls = []
    for label in label_mappings:
        cls.append(len(label))
    model = Model(backbone=args.backbone, num_classes=cls, num_heads=len(args.label_columns),
                  concat_gvf=args.global_video_features is not None)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    if args.backbone == 'i3d':
        backbone_params = chain(model.features.Conv3d_1a_7x7.parameters(),
                                model.features.Conv3d_2b_1x1.parameters(),
                                model.features.Conv3d_2c_3x3.parameters(),
                                model.features.Mixed_3b.parameters(),
                                model.features.Mixed_3c.parameters(),
                                model.features.Mixed_4b.parameters(),
                                model.features.Mixed_4c.parameters(),
                                model.features.Mixed_4d.parameters(),
                                model.features.Mixed_4e.parameters(),
                                model.features.Mixed_4f.parameters(),
                                model.features.Mixed_5b.parameters(),
                                model.features.Mixed_5c.parameters())
    else:
        backbone_params = chain(model.features.layer1.parameters(),
                                model.features.layer2.parameters(),
                                model.features.layer3.parameters(),
                                model.features.layer4.parameters())

    if len(args.label_columns) == 1:
        fc_params = model.fc.parameters()
    else:
        fc_params = chain(model.fc1.parameters(), model.fc2.parameters())

    if args.backbone == 'i3d':
        params = [
            {'params': backbone_params, 'lr': args.backbone_lr, 'name': 'backbone'},
            {'params': fc_params, 'lr': args.fc_lr, 'name': 'fc'}
        ]
    else:
        params = [
            {'params': model.features.stem.parameters(), 'lr': 0, 'name': 'stem'},
            {'params': backbone_params, 'lr': args.backbone_lr, 'name': 'backbone'},
            {'params': fc_params, 'lr': args.fc_lr, 'name': 'fc'}
        ]
    optimizer = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay)
    warmup_iters = args.lr_warmup_epochs * len(data_loader_train)
    lr_milestones = []
    for m in args.lr_milestones:
        lr_milestones.append(len(data_loader_train) * m)
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
                                     warmup_iters=warmup_iters, warmup_factor=1e-5)
    if args.resume:
        print(f'Resuming from checkpoint {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    print('START TRAINING')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model=model, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler,
                        data_loader=data_loader_train, device=device, epoch=epoch, print_freq=args.print_freq,
                        label_columns=args.label_columns, loss_alphas=args.loss_alphas)
        if args.output_dir:
            checkpoint = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'epoch': epoch,
                          'args': args}
            torch.save(checkpoint, os.path.join(args.output_dir, f'epoch_{epoch}.pth'))
            torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))
        evaluate(model=model, criterion=criterion, data_loader=data_loader_valid, device=device, epoch=epoch,
                 print_freq=args.print_freq, label_columns=args.label_columns, loss_alphas=args.loss_alphas,
                 output_dir=args.output_dir)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')


if __name__ == '__main__':
    from opts import parse_args

    args = parse_args()
    main(args)
