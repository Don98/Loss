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

print('CUDA available: {}'.format(torch.cuda.is_available()))
import Don
import os
def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a cnn3 network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--model_path', help='Path to model', type=str)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=101)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--use_model', help='Use model or not?', type=int, default=0)
    
    parser = parser.parse_args(args)

    # Create the data loaders
    
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer([0,0])]),part = 1)
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer([0,0])]),part = 1)
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
    if parser.use_model == 0:
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
    
        use_gpu = True

        if use_gpu:
            if torch.cuda.is_available():
                cnn3 = cnn3.cuda()

        if torch.cuda.is_available():
            cnn3 = torch.nn.DataParallel(cnn3).cuda()
        else:
            cnn3 = torch.nn.DataParallel(cnn3)
    else:
            cnn3 = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)

            use_gpu = True
            
            if use_gpu:
                if torch.cuda.is_available():
                    cnn3 = cnn3.cuda()

            cnn3 = torch.nn.DataParallel(cnn3).cuda()
            model_dict = torch.load(parser.model_path).module.state_dict()
            cnn3.module.load_state_dict(model_dict)

    cnn3.training = True

    optimizer = optim.Adam(cnn3.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)
    
    cnn3.train()
    cnn3.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    # name = parser.model_path[:29]
    # the_num = int(parser.model_path[29:-3])
    pos = parser.model_path.index("\\")
    model_name = parser.model_path[:pos]
    name = parser.model_path[:pos * 2 + 1]
    print(parser.model_path[:pos*2+1],parser.model_path[pos*2+1:-3])
    the_num = int(parser.model_path[pos*2+1:-3])
    os.chdir(model_name)
    for epoch_num in range(parser.epochs):
        the_path = "Epoch" + str(the_num + epoch_num + 1)
        if not os.path.isdir(the_path):
            os.mkdir(the_path)
        os.chdir(the_path)
        cnn3.train()
        cnn3.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = cnn3([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = cnn3([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(cnn3.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue
        os.chdir("../")
        # if parser.dataset == 'coco':

            # print('Evaluating dataset')

            # coco_eval.evaluate_coco(dataset_val, cnn3)

        # elif parser.dataset == 'csv' and parser.csv_val is not None:

            # print('Evaluating dataset')

            # mAP = csv_eval.evaluate(dataset_val, cnn3)
            
        # elif parser.dataset == 'voc':


            # print('Evaluating dataset')

            # mAP = voc_eval.evaluate(dataset_val, cnn3)
        scheduler.step(np.mean(epoch_loss))

        # torch.save(cnn3.module, '{}_cnn3_{}.pt'.format(parser.dataset, epoch_num))
        # torch.save(cnn3, 'lh_model_final' + str( 5 + epoch_num + 1 ) + '.pt')
        torch.save(cnn3, model_name + str( the_num + epoch_num + 1 ) + '.pt')
        print("Save as : ",model_name + str( the_num + epoch_num + 1 ) + '.pt')
        
    cnn3.eval()

    torch.save(cnn3, model_name + str( the_num + parser.epochs ) + '.pt')
    print("Save as : ",model_name + str( the_num + parser.epochs ) + '.pt')
    Don.Mess()

if __name__ == '__main__':
    main()
