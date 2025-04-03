import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import math

def column_sort(matrix):
    # Sort each column
    matrix = torch.sort(matrix, dim=1).values  # Sort along the second dimension
    
    # Transpose (permute dimensions), sort each column again, and transpose back
    matrix = torch.sort(matrix.permute(0, 2, 1), dim=1).values.permute(0, 2, 1)
    
    # Final column sort
    matrix = torch.sort(matrix, dim=1).values
    
    return matrix

class BasicSortNet(nn.Module):
    def __init__(self, input_size):
        super(BasicSortNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, input_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RecursiveSortNet(nn.Module):
    def __init__(self, input_size, recursion_depth=1):
        super(RecursiveSortNet, self).__init__()
        self.recursion_depth = recursion_depth
        self.basic_sort_net = BasicSortNet(input_size)
    
    def forward(self, x):
        if self.recursion_depth == 0:
            return self.basic_sort_net(x)
        else:
            # Reshape to 2D matrix for column sorting
            n = int(math.sqrt(x.size(1)))
            if n * n != x.size(1):
                raise ValueError("Input size must be a perfect square for reshaping")
            x_reshaped = x.view(-1, n, n)
            
            # Apply column sort
            x_sorted = column_sort(x_reshaped)
            
            # Flatten and apply basic sort network
            x_flattened = x_sorted.view(-1, x.size(1))
            return self.basic_sort_net(x_flattened)

class RandomDataset(Dataset):
    def __getitem__(self, index):
        size = 1024  # Use a perfect square size (e.g., 1024 = 32 * 32)
        input = torch.randint(0, 100, (size,), dtype=torch.float32)
        sorted_values = torch.sort(input).values
        label = sorted_values.to(dtype=torch.float32)
        return input, label

    def __len__(self):
        return 10000

class Trainer:
    def __init__(self, model):
        self.model = model
        self.dataset = RandomDataset()
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Initial learning rate
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.8)  # Decrease every 5 epochs
  
    def train(self):
        num_epochs = 200
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (input, label) in enumerate(self.dataloader):
                output = self.model(input)
                loss = self.criterion(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'Epoch {epoch+1}, Iteration {i+1}, Loss: {running_loss / 100}')
                    running_loss = 0.0
            self.scheduler.step()  # Step the learning rate scheduler
        torch.save(self.model.state_dict(), 'trained_model.pth')

if __name__ == '__main__':
    input_size = 1024  # Use the updated size
    recursion_depth = 2
    model = RecursiveSortNet(input_size, recursion_depth)
    trainer = Trainer(model)
    trainer.train()
