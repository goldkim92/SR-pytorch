from __future__ import print_function

from tqdm import tqdm
from math import log10

import torch
import torch.backends.cudnn as cudnn

from SRCNN.model import Net


class SRCNNTrainer(object):
    def __init__(self, args, train_loader, test_loader):
        super(SRCNNTrainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.lr = args.lr
        self.epochs = args.epochs
        self.seed = args.seed
        self.upscale_factor = args.upscale_factor
        self.train_loader = train_loader
        self.test_loader = test_loader

    def build_model(self):
        self.model = Net(num_channels=3, 
                         base_filter=64, 
                         upscale_factor=self.upscale_factor).to(self.device)
        self.model.weight_init(mean=0.0, std=0.01)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)
        print(self.model)
        print('===========================================================')
        
        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)

    def save_model(self):
        model_out_path = "model_srcnn.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in tqdm(enumerate(self.train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        print("    Average Loss: {:.4f}".format(train_loss / len(self.train_loader)))

    def test(self):
        self.model.eval()
        avg_psnr = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                mse = self.criterion(self.model(data), target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.test_loader)))

    def run(self):
        self.build_model()
        for epoch in range(1, self.epochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.scheduler.step(epoch)
            if epoch == self.epochs:
                self.save_model()
