import argparse
import os

from torch.utils.data import DataLoader

from SRCNN.solver import SRCNNTrainer

from dataloader import get_dataset


# ===========================================================
# settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')

# hyper-parameters
parser.add_argument('--gpu_number', type=str, default='2')
parser.add_argument('--bs', type=int, default=32, help='training batch size')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# model configuration
parser.add_argument('--model', '-m', type=str, default='srcnn', help='choose which model to use')
parser.add_argument('--upscale_factor', '-uf',  type=int, default=3, help="super resolution upscale factor")
parser.add_argument('--patch_size', '-ps',  type=int, default=128, help="target size")


args = parser.parse_args()

# setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number


# ===========================================================
# main
# ===========================================================
def main():
    print('===> Loading datasets')
    train_set = get_dataset(args.patch_size, args.upscale_factor, phase='train')
    train_loader = DataLoader(dataset=train_set, 
                              batch_size=args.bs, 
                              shuffle=True)
    
    test_set = get_dataset(0, args.upscale_factor, phase='test') # 0 has no meaning
    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             shuffle=False)
    
    if args.model == 'srcnn':
        model = SRCNNTrainer(args, train_loader, test_loader)
    elif args.model == 'srdcn':
        model = SRDCNTrainer(args, train_loader, test_loader)
    else:
        raise Exception("the model does not exist")

    model.run()


if __name__ == '__main__':
    main()
