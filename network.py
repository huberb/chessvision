import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary
from dataloader import tensor_to_board


def create_batches(images, labels, batch_size, device='cuda'):
    for i in range(0, len(images), batch_size):
        batch_x = images[i:i + batch_size]
        batch_y = labels[i:i + batch_size]
        yield batch_x.to(device), batch_y.to(device)


class Network(nn.Module):
    def __init__(self, depth=128, device='cuda'):
        super(Network, self).__init__()

        self.model = nn.Sequential(
                nn.Conv2d(1, depth, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(depth),
                nn.ReLU(),

                nn.Conv2d(depth, depth, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(depth),
                nn.ReLU(),

                nn.Conv2d(depth, depth, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(depth),
                nn.ReLU(),

                nn.Conv2d(depth, depth, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(depth),
                nn.ReLU(),

                nn.Conv2d(depth, depth, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(depth, 13, kernel_size=3, padding=1),
                nn.Softmax(dim=1),
                ).to(device)

        self.optim = optim.Adam(self.model.parameters(), lr=0.0002)
        self.mse = nn.MSELoss()
        summary(self.model, (1, 256, 256))

    def forward(self, x):
        output = self.model(x)
        return output

    def train(self, images, labels, batch_size):
        summed_loss = 0
        for batch_x, batch_y in tqdm(
                create_batches(images, labels, batch_size),
                total=len(images) // batch_size,
                unit_scale=batch_size
                ):
            self.model.zero_grad()
            output = self.model(batch_x)
            loss = self.mse(output, batch_y)
            loss.backward()
            self.optim.step()
            summed_loss += loss.item()
        print(f"epoch finished. loss: {summed_loss}")

        board_predicted = tensor_to_board(output[0].detach().cpu().numpy())
        print("output:")
        print(board_predicted)
        board_real = tensor_to_board(batch_y[0].cpu().numpy())
        print("label:")
        print(board_real)

    def test(self, images, labels, batch_size):
        self.model.eval()
        outputs = torch.empty((len(labels), *labels.shape[1:]))
        index = 0
        for batch_x, batch_y in tqdm(
                create_batches(images, labels, batch_size),
                total=len(images) // batch_size,
                unit_scale=batch_size
                ):
            output = self.model(batch_x).detach()
            outputs[index:index + batch_size] = output
            index += batch_size

        hitrate = self.compare_boards(outputs, labels)

        print(f"piece accuracy: {hitrate}")
        self.model.train()

    def compare_boards(self, outputs, labels):
        hitrate = []
        board_size = 8
        for output, label in zip(outputs, labels):
            output = torch.einsum('ijk->jki', output)
            label = torch.einsum('ijk->jki', label)
            for x in range(board_size):
                for y in range(board_size):
                    real_piece = torch.argmax(label[x][y])
                    predicted_piece = torch.argmax(output[x][y])
                    hitrate.append(real_piece == predicted_piece)
        return sum(hitrate) / len(hitrate)
