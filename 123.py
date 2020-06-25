import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from CNN3 import model
from CNN3.dataloader import CocoDataset, CSVDataset, VocDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from CNN3 import coco_eval
from CNN3 import csv_eval
from CNN3 import voc_eval

assert torch.__version__.split('.')[0] == '1'
import Don
print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    model_name = "hh_model_final"
    # model_name = "low80h_model_final"
    parser = argparse.ArgumentParser(description='Simple training script for training a cnn3 network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=101)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    
    parser = parser.parse_args(args)

    # Create the data loaders
    
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    # transform=transforms.Compose([Normalizer(), Augmenter(), Resizer([460,640])]),part = 1)
                                  transform=transforms.Compose([Normalizer(), Resizer([350,500])]),part = 1)
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer([460,640])]),part = 1)
                                  # transform=transforms.Compose([Normalizer(), Resizer([350,500])]),part = 1)

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'voc':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = VocDataset(parser.coco_path, set_name='2007',name = "train2",
                                    # transform=transforms.Compose([Normalizer(), Augmenter(), Resizer([480,600])]))
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer([800,1000])]))
                                    # transform=transforms.Compose([Normalizer(), Augmenter()]))
        dataset_val = VocDataset(parser.coco_path, set_name='2007',name = "trainval2",
                                  transform=transforms.Compose([Normalizer(), Resizer([800,1000])]))
                                  # transform=transforms.Compose([Normalizer(), Resizer([350,500])]),part = 1)
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=1, collate_fn=collater, batch_sampler=sampler)
    
    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)
    
    # Create the model
    if parser.depth == 18:
        cnn3 = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        cnn3 = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        cnn3 = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        cnn3 = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        cnn3 = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    # print(dataset_train.image_ids[0])
    num = 0
    for i in dataset_train: 
        # print(dataset_train.image_ids[num])
        print(i["annot"])
        num += 1
        if(num == 5):
            break
    
    Don.Mess()

if __name__ == '__main__':
    main()
