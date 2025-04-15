import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Covolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      bias=False,
                      ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=16,
                          out_channels=16,
                          kernel_size=5,
                          stride=1,
                          padding=2,
                          bias=False,
                          ),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            )
            for _ in range(2)
        ])

        #Fully connected layer
        self.out = nn.Linear(16*3*3, 10)

    def forward(self, x):
        for conv in [self.conv1, self.conv[0], self.conv[1]]:
            x = conv(x)
        # flatten the output of conv2 to (batch_size, 16*3*3)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
