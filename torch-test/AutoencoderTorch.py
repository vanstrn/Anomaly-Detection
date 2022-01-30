import time
import torch
import torchvision
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader

class MNIST_RECON():
    def __init__(self):
        (x_train, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

        self.trainData = {
            "image":np.expand_dims(RGBtoNORM(x_train).astype(np.float32),axis=-1)
        }
        self.testData = {
            "image":np.expand_dims(RGBtoNORM(x_test).astype(np.float32),axis=-1),
            "label":y_test.astype(np.float32),
        }

        self.inputSpec = {"image":[28,28,1]}
        self.outputSpec = {"image_size":[28,28,1]}



class Autoencoder(nn.Module):
    def __init__(self, latentSize):
        super(Autoencoder, self).__init__()

        ## Encoder Layers
        self.ActivationEnc = nn.ReLU(inplace=True,)

        self.conv1 = nn.Conv2d(1, 64, 5, stride=2, padding=0, bias=True)
        self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=0, bias=True)
        self.drop1 = nn.Dropout(0.3)
        self.Dense1 = nn.Linear(128*4*4, 256)
        self.Dense2 = nn.Linear(256, latentSize)

        ## Decoder Layers
        self.ActivationDec = nn.LeakyReLU(inplace=True, negative_slope=0.2)

        self.Dense3 = nn.Linear(latentSize, 256)
        self.norm1 = nn.BatchNorm1d(256)
        self.Dense4 = nn.Linear(256,256*7*7)
        self.norm2 = nn.BatchNorm1d(256*7*7)
        self.convT1 = nn.ConvTranspose2d(256, 128, 5, stride=1, padding=2, bias=False)
        self.norm3 = nn.BatchNorm2d(128)
        self.convT2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.norm4 = nn.BatchNorm2d(64)
        self.convT3 = nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False)

    def forward(self, x):
        z = self.conv1(x)
        z = self.ActivationEnc(z)
        z = self.drop1(z)
        z = self.conv2(z)
        z = self.ActivationEnc(z)
        z = self.drop1(z)
        z=z.view(z.size(0), 128*4*4)
        z = self.Dense1(z)
        z = self.ActivationEnc(z)
        latent = self.Dense2(z)

        z = self.Dense3(latent)
        z = self.norm1(z)
        z = self.ActivationDec(z)
        z = self.Dense4(z)
        z = self.norm2(z)
        z = self.ActivationDec(z)
        z=z.view(z.size(0), 256, 7, 7)
        z = self.convT1(z)
        z = self.norm3(z)
        z = self.ActivationDec(z)
        z = self.convT2(z)
        z = self.norm4(z)
        z = self.ActivationDec(z)
        out = self.convT3(z)

        return out, latent

def main():
    device = 1
    autoencoder = Autoencoder(8).cuda()

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
    data_train = (train_set.data/ 127.5) - 1
    data_train = torch.unsqueeze(data_train, 1)
    data = DataLoader(data_train, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.00005 )
    timeStart = time.time()
    for i in range(25):
        timeEpochStart = time.time()
        for batch in data:
            inImages = batch.cuda()
            autoencoder.train()
            autoencoder.zero_grad()

            generatedImage, _ = autoencoder(inImages)
            loss = (generatedImage - inImages).pow(2).mean()

            loss.backward()
            optimizer.step()
        print("Epoch {}: ||Time: {}".format(i,time.time()-timeEpochStart))
    print("Final runtime: {}".format(time.time()-timeStart))


if __name__ == "__main__":
    main()
