import argparse
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torchvision import transforms

from CNN3 import model
from CNN3.dataloader import VocDataset, Resizer, Normalizer
from CNN3 import voc_eval

assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a CNN3 network.')

    parser.add_argument('--voc_path', help='Path to VOC directory')
    parser.add_argument('--model_path', help='Path to model', type=str)

    parser = parser.parse_args(args)

    # dataset_val = VocDataset(parser.voc_path, set_name='2007',name = "trainval2",
    dataset_val = VocDataset(parser.voc_path, set_name='2007',name = "val",
                                  # transform=transforms.Compose([Normalizer(), Resizer([800,1000])]))
                                  transform=transforms.Compose([Normalizer(), Resizer([0,0])]))
    # Create the model
    cnn3 = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)

    use_gpu = True
    
    if use_gpu:
        if torch.cuda.is_available():
            cnn3 = cnn3.cuda()

    cnn3 = torch.nn.DataParallel(cnn3).cuda()
    model_dict = torch.load(parser.model_path).module.state_dict()
    cnn3.module.load_state_dict(model_dict)

    cnn3.training = False
    cnn3.eval()
    cnn3.module.freeze_bn()
    print(parser.model_path[:-4])
    voc_eval.evaluate(dataset_val, cnn3,model_path = parser.model_path[:-4])

if __name__ == '__main__':
    main()
