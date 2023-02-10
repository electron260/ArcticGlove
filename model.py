
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=5,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=2,
                stride=1,
                padding=1

            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(256, 50)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
       
        x = self.conv1(x)
      
   
        x = torch.flatten(x, 1)
    
   
  
        x =self.fc1(x)

   
    
        logits = self.fc2(x)
      
   
    
        
        
        

        return logits
