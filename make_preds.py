import argparse
import os
import sys
from PIL import Image
import torch
import torch.nn.parallel
import torch.nn.functional as nn_func
import torchvision.transforms as transforms
import torch.utils.data.distributed
from tqdm import tqdm
import torch.nn as nn
# Replace this with your data loader and edit the calls accordingly
# from data_loader import EvalDataset
from efficientnet_pytorch import EfficientNet, EfficientNetTH
efficientnet_th = lambda num_classes: EfficientNetTH.from_name('efficientnet-b4', num_classes=num_classes)

def get_parser():
    """"Defines the command line arguments"""
    parser = argparse.ArgumentParser(description='Open World Vision')
    parser.add_argument('--input_file', required=True,
                        help='path to a .txt/.csv file containing paths of input images in first column of each row. '
                             '\',\' will be used as a delimiter if a csv is provided. In text format, each row should'
                             ' only contain the path of an image.')
    parser.add_argument('--dataroot', required=True, help='dataroot')
    parser.add_argument('--out_dir', required=True,
                        help='directory to be used to save the results. We will save a \',\' separated csv which will'
                             ' be named by the next argument: <exp_name> ')
    parser.add_argument('--exp_name', required=True,
                        help='unique name for this run of the evaluation')
    parser.add_argument('--model_path', required=True,
                        help='path to model file')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--accimage', action='store_true',
                        help='use if accimage module is available on the system and you want to use it')
    return parser

class ModuleWrapper(nn.Module):
    def __init__(self, model):
        super(ModuleWrapper, self).__init__()
        self.module = model

    def forward(self, x):
        return self.module(x)

def load_images(data_file):
    with open(data_file) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content][1:]
    return [x.split(',')[0].split('/')[1] for x in content]

class EvalImagesDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, imgs_list, transforms):
        super(EvalImagesDataset, self).__init__()
        self.dataroot = dataroot
        self.images = imgs_list
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        path = os.path.join(self.dataroot, 'val', self.images[i])
        img = Image.open(path).convert('RGB')
        return i, self.transforms(img)



def run(model_path, data_file, dataroot, exp_name, out_dir, accimage=False, batch_size=32, workers=4):
    """Runs the model on given data and saves the class probabilities

    Args:
        model_path (str): path to pytorch model file
        data_file (str): path to txt/csv file containing input images. If txt each line should only contain the path of
            an image. If csv,  1st column should have image paths
        exp_name (str): unique name for the experiment. Will be used to save output
        out_dir (str): path to dump output files
        accimage (bool): whether to use accimage loader. If calling this function outside this module, please make sure
            that accimage is importable in your python env
        batch_size (int): batchsize for model
        workers (int): no. of workers to be used in dataloader
    """
    try:
        checkpoint = torch.load(model_path)
        #model = checkpoint['model']
        # model = ModuleWrapper(resnet18(num_classes=413)).cuda()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(efficientnet_th(num_classes=413).cuda())
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(checkpoint["state_dict"])
        # model = KPlus1Wrapper(model, 0)
        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            # Replace this with your data-loader
            # test_set = EvalDataset(
            #     data_file=data_file,
            #     accimage=accimage,
            #     transform=transforms.Compose([
            #         transforms.Resize(256),
            #         transforms.CenterCrop(224),
            #         transforms.ToTensor(),
            #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                              std=[0.229, 0.224, 0.225]),
            #         ]),
            #     header=True,
            # )
            image_names = load_images(data_file)
            test_set = EvalImagesDataset(
                dataroot,
                image_names,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
            )

            # img_path_list = test_set.data_list
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=True)

            img_idx_list = list()
            output_list = list()
            for img_idx, images in tqdm(test_loader):
                images = images.cuda()
                output, ood_out = model(images)
                output = torch.cat((ood_out, output), dim=1)
                # Adjust these according to your model
                # output = nn_func.softmax(output[0], 1)
                # output = nn_func.softmax(output[:, :413], 1).cpu()
                # zero_vec = torch.zeros((output.shape[0], 1))
                output = nn_func.softmax(output, dim=1)
                # ood_probs = nn_func.softmax(ood_out, dim=1)[:, 0]
                # zero_vec = ood_probs.unsqueeze(-1)
                # print(ood_probs)
                # zero_vec[zero_vec < (1-1e-14)] = 0
                # zero_vec[zero_vec < 0.5] = 0.
                # output = output / output.sum(dim=1, keepdim=True)
                # output = output * (1 - zero_vec)
                #output = torch.cat([zero_vec, output], dim=1)
                #output = output / output.sum(dim=1, keepdim=True)
                img_idx_list.append(img_idx)
                output_list.append(output)
            img_idx_list = torch.cat(img_idx_list, 0)
            output_list = torch.cat(output_list, 0)

            output_list = output_list[img_idx_list]
            lines = list()
            for i in range(img_idx_list.shape[0]):
                line = [str(x) for x in output_list[i].tolist()]
                lines.append(','.join([image_names[i]] + line))
            # for i, img_idx in enumerate(img_idx_list):
            #     line = [str(x) for x in output_list[i].tolist()]
            #     lines.append(','.join([img_path_list[img_idx]] + line))
            with open(os.path.join(out_dir, f'{exp_name}.csv'), 'w') as f:
                f.write('\n'.join(lines))
    # except FileNotFoundError:
    #     print(f'Could not find the model file at {model_path}')
    # except KeyError:
    #     print(f'Saved model does not have expected format. We expect the checkpoint to have \'model\' and '
    #           f'\'state_dict\' keys')
    except Exception as e:
        raise e


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.accimage:
        try:
            import accimage
        except ModuleNotFoundError:
            print('You opted for using accimage but we are unable to import it. Process will be terminated.')
            sys.exit()

    run(args.model_path, args.input_file, args.dataroot, args.exp_name, args.out_dir, args.accimage, args.batch_size, args.workers)


if __name__ == '__main__':
    main()
