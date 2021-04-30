import os
import cv2
import numpy as np
import torch

# Constants
pieces = ['.']
pieces.extend(['P', 'R', 'N', 'B', 'Q', 'K'])
pieces.extend(['p', 'r', 'n', 'b', 'q', 'k'])
resized_board_width = 256
board_size = 8


def board_to_tensor(board_path):
    tensor = np.zeros((board_size, board_size, len(pieces)))
    with open(board_path, 'r') as f:
        for x, line in enumerate(f.readlines()):
            line = line.replace("\n", "")
            for y, piece in enumerate(line.split(" ")):
                tensor[x][y][pieces.index(piece)] = 1
    return np.einsum('ijk->kij', tensor)


def tensor_to_board(tensor):
    tensor = np.einsum('ijk->jki', tensor)
    board = ""
    for x in range(board_size):
        for y in range(board_size):
            piece_index = np.argmax(tensor[x][y])
            piece = pieces[piece_index]
            board += piece
            if y != board_size - 1:
                board += " "
        board += "\n"
    return board


def load_image(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (resized_board_width, resized_board_width))
    img = np.expand_dims(img, axis=2)
    img = np.einsum('ijk->kij', img)
    return (img / 255).astype(np.float32)


def load_data(path="data", count=None):
    label_folder = f"{path}/label"
    img_folder = f"{path}/images"
    if count is None:
        count = len(os.listdir(img_folder))
    labels = np.empty((count, len(pieces), board_size, board_size))
    images = np.empty((count, 1, resized_board_width, resized_board_width))
    for i in range(count):
        label_path = f"{label_folder}/{i}.md"
        img_path = f"{img_folder}/{i}.png"
        label = board_to_tensor(label_path)
        image = load_image(img_path)
        labels[i] = label
        images[i] = image
    return torch.Tensor(images), torch.Tensor(labels)


def test():
    label = board_to_tensor("./data/label/0.md")
    board = tensor_to_board(label)
    print(board)
