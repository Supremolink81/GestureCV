import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0.0)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0.0)
    
        self.fc1 = nn.Linear(in_features=12544, out_features=1028)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)

        self.fc2 = nn.Linear(in_features=1028, out_features=120)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

        self.fc3 = nn.Linear(in_features=120, out_features=84)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)

        self.fc4 = nn.Linear(in_features=84, out_features=5)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0.0)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Implement the forward pass of the model
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.softmax(self.fc4(x))