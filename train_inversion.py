import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import datasets, transforms


class Classifier(nn.Module):
    def __init__(self, nc, ndf, nz):
        super(Classifier, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.nz = nz

        self.encoder = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*4) x 4 x 4
        )

        self.fc = nn.Sequential(
            nn.Linear(ndf * 4 * 4 * 4, nz * 5),
            nn.Dropout(0.5),
            nn.Linear(nz * 5, nz),
        )

    def forward(self, x, release=False):

        x = x.view(-1, 1, 32, 32)
        x = self.encoder(x)
        x = x.view(-1, self.ndf * 4 * 4 * 4)
        x = self.fc(x)
        
        if release:
            return F.softmax(x, dim=1)
        else:
            return x


class InversionNet(nn.Module):
    def __init__(self, nz, ngf, nc, truncation, c):
        super(InversionNet, self).__init__()
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.truncation = truncation
        self.c = c
        
        self.decoder = nn.Sequential(
            # input is Z
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 32 x 32
        )
        
    def truncation_vector(self, x):
        top_k, indices = torch.topk(x, self.truncation)
        top_k = torch.clamp(torch.log(top_k), min=-1000) + self.c
        top_k_min = top_k.min(1, keepdim=True)[0]
        top_k = top_k + F.relu(-top_k_min)
        # x = torch.zeros(len(x), self.nz).scatter_(1, indices, top_k)
        x = torch.zeros(len(x), self.nz).cuda().scatter_(1, indices, top_k)

        return x

    def forward(self, x):
        x = self.truncation_vector(x)
        x = x.view(-1, self.nz, 1, 1)
        x = self.decoder(x)
        x = x.view(-1, 1, 32, 32)
        return x
    
    
def train(classifier, inversion_model, device, inversion_train_loader, optimizer, epoch, f, k_top, log_interval):
    classifier.eval()
    inversion_model.train()
    for batch_idx, (data, target) in enumerate(inversion_train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            prediction = classifier(data, release=True)
            prob, pred_label = torch.max(prediction, dim=1)
            _, pred_label_pseudo = torch.topk(prediction, k_top, dim=1)
        reconstruction = inversion_model(prediction)
        loss = F.mse_loss(reconstruction, data)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                 len(inversion_train_loader.dataset), loss.item()),
                  file=f, flush=True)
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                 len(inversion_train_loader.dataset), loss.item()),)


def test(classifier, inversion_model, device, inversion_test_loader, epoch, path_out, train_loader, f, plot=False):
    classifier.eval()
    inversion_model.eval()
    mse_loss = 0
    with torch.no_grad():
        for data, target in inversion_test_loader:
            data, target = data.to(device), target.to(device)
            prediction = classifier(data, release=True)
            reconstruction = inversion_model(prediction)
            mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()

            if plot:
                truth = data[0:512]
                inverse = reconstruction[0:512]
                out = torch.cat((inverse, truth))
                for i in range(16):
                    out[i * 32:i * 32 + 16] = inverse[i * 16:i * 16 + 16]
                    out[i * 32 + 16:i * 32 + 32] = truth[i * 16:i * 16 + 16]
                vutils.save_image(out, path_out + 'recon_round_attack.png', nrow=32, normalize=True)
                plot = False

    mse_loss /= len(inversion_test_loader.dataset) * 32 * 32
    # logger.info('\nTest inversion model on {} epoch: Average MSE loss: {:.6f}\n'.format(epoch, mse_loss))
    print('\nTest inversion model on {} epoch (Test set): Average MSE loss: {:.6f}\n'.format(epoch, mse_loss),
          file=f, flush=True)
    print('\nTest inversion model on {} epoch (Test set): Average MSE loss: {:.6f}\n'.format(epoch, mse_loss))

    mse_loss_pri = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            prediction = classifier(data, release=True)
            reconstruction = inversion_model(prediction)
            mse_loss_pri += F.mse_loss(reconstruction, data, reduction='sum').item()

            if plot:
                truth = data[0:512]
                inverse = reconstruction[0:512]
                out = torch.cat((inverse, truth))
                for i in range(16):
                    out[i * 64:i * 64 + 32] = inverse[i * 32:i * 32 + 32]
                    out[i * 64 + 32:i * 64 + 64] = truth[i * 32:i * 32 + 32]
                vutils.save_image(out, path_out + 'recon_round_private.png', nrow=32, normalize=True)
                plot = False

    mse_loss_pri /= len(train_loader.dataset) * 32 * 32
    # logger.info('\nTest inversion model on {} epoch: Average MSE loss [Private]: {:.6f}\n'.format(epoch, mse_loss_pri))
    print('\nTest inversion model on {} epoch: Average MSE loss (Train set): {:.6f}\n'.format(epoch, mse_loss_pri),
          file=f, flush=True)
    print('\nTest inversion model on {} epoch: Average MSE loss (Train set): {:.6f}\n'.format(epoch, mse_loss_pri))

    return mse_loss


if __name__ == '__main__':
    f = open('out/train_inversion_model.txt', 'a')
    batch_size = 512
    epochs = 500
    learning_rate = 0.001
    early_stop = 20
    image_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUse device: {}".format("cuda" if torch.cuda.is_available() else "cpu"), file=f, flush=True)
    
    train_data = datasets.MNIST('data', train=True, download=True,
                                transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))]))
    
    test_data = datasets.MNIST('data', train=False, download=True,
                                transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))]))
    
    all_data = torch.utils.data.ConcatDataset([train_data, test_data])
    train_set, test_set = torch.utils.data.random_split(all_data, [0.5, 0.5], torch.Generator().manual_seed(42))
    
    ## split test set, 50% as training set of inversion model
    train_inversion_set, test_inversion_set = torch.utils.data.random_split(test_set, [0.5, 0.5], torch.Generator().manual_seed(42))
    
    inversion_train_loader = torch.utils.data.DataLoader(train_inversion_set, batch_size=batch_size, shuffle=True)
    inversion_test_loader = torch.utils.data.DataLoader(test_inversion_set, batch_size=batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    nc = 1
    ngf = 128
    ndf = 128
    nz = 10
    c = 50
    truncation = 10
    
    classifier = Classifier(nc=nc, ndf=ndf, nz=nz).to(device)
    path = 'out/tmp_target_classifier_s1_20epoch.pt'
    
    if torch.cuda.is_available():
        classifier.load_state_dict(torch.load(path, weights_only=True))
    else:
        classifier.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    inversion_model = InversionNet(nz=nz, ngf=ngf, nc=nc, truncation=truncation, c=c).to(device)
    optimizer = optim.Adam(inversion_model.parameters(), lr=learning_rate, betas=(0.5, 0.999), amsgrad=True)
    path_out = 'out/inversion_target20epoch_'
    best_recon_loss = 999
    early_stop_label = 0

    for epoch in range(1, epochs + 1):
        train(classifier, inversion_model, device, inversion_train_loader, optimizer, epoch, f, k_top=2, log_interval=20)
        recon_loss = test(classifier, inversion_model, device, inversion_test_loader, epoch, path_out, train_loader, f)
        
        if recon_loss < best_recon_loss:
            best_recon_loss = recon_loss
            state = {
                'epoch': epoch,
                'model': inversion_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_recon_loss': best_recon_loss
            }
            torch.save(state, path_out + 'inversion_model.pth')
            # shutil.copyfile('Inversion_Models/' + args.path_out + 'recon_round_attack.png',
            #                 'Inversion_Models/' + args.path_out + 'best.png')
            early_stop_label = 0
        else:
            early_stop_label += 1
            if early_stop_label == early_stop:
                print('\nThe best test inversion model on {} epoch: Average MSE loss: {:.6f}\n'.format(epoch,
                                                                                                       best_recon_loss))
                break
