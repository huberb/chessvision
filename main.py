import argparse
import torch
from sklearn.model_selection import train_test_split
from network import Network
from pathlib import Path
from dataloader import load_data, load_image, tensor_to_board


def train(dataset_size=None, epochs=3, batch_size=32):
    model = Network()
    images, labels = load_data(count=dataset_size)

    train_x, test_x, train_y, test_y = \
        train_test_split(images, labels, test_size=0.1)

    for e in range(epochs):
        model.train(train_x, train_y, batch_size=batch_size)
        model.test(test_x, test_y, batch_size=batch_size)

    Path("saved_models").mkdir(exist_ok=True)
    torch.save(model.model, "./saved_models/model.pt")


def predict(path):
    model = torch.load("./saved_models/model.pt").to('cpu')
    img = load_image(path)
    img = torch.Tensor([img])
    output = model(img)[0].detach().numpy()
    print("output:")
    print(tensor_to_board(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store',
                        type=str, help='path to test png')
    args = parser.parse_args()

    if args.predict:
        predict(args.predict)
    else:
        torch.manual_seed(42)
        train()
