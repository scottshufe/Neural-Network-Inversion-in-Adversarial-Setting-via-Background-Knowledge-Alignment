import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


def train(model, device, train_loader, optimizer, criterion, epoch, f):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), file=f, flush=True)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion, f):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)), file=f, flush=True)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    f = open('out/train_target_model_s1.txt', 'a')
    batch_size = 512
    epochs = 30
    image_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUse device: {}".format("cuda" if torch.cuda.is_available() else "cpu"), file=f, flush=True)
    print("\nUse device: {}".format("cuda" if torch.cuda.is_available() else "cpu"))
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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    nz = 10
    nc = 1
    ndf = 128
    
    model = Classifier(nc=nc, ndf=ndf, nz=nz).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999), amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch, f)
        if epoch % 10 == 0:
            test(model, device, test_loader, criterion, f)
            path = 'out/tmp_target_classifier_s1_{}epoch.pt'.format(epoch)
            torch.save(model.state_dict(), path)


if __name__ == '__main__':
    main()
