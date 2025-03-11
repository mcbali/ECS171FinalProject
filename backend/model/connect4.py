import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


class ConnectFourDeepNet(nn.Module):
    def __init__(self):
        super(ConnectFourDeepNet, self).__init__()
        self.fc1 = nn.Linear(42, 500)
        self.fc2 = nn.Linear(500, 20)
        self.fc3 = nn.Linear(20, 500)
        self.fc4 = nn.Linear(500, 3)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.log_softmax(x, dim = 0)
        return x


def convert_to_board_state(col1, col2, col3, col4, col5, col6, col7):
    board_state = []
    for i in range(6):
        board_state.append(col1[i])
        board_state.append(col2[i])
        board_state.append(col3[i])
        board_state.append(col4[i])
        board_state.append(col5[i])
        board_state.append(col6[i])
        board_state.append(col7[i])
    return board_state

def find_best_move(board_state, difficulty):
    model_path = ""

    if difficulty == "hard":
        model_path = resource_path("hard_model.pth")
    elif difficulty == "medium":
        model_path = resource_path("medium_model.pth")
    else:
        model_path = resource_path("easy_model.pth")

    model = torch.load(model_path)

    model.eval()

    with torch.no_grad():
        col1 = []
        col2 = []
        col3 = []
        col4 = []
        col5 = [] 
        col6 = []
        col7 = []  
        i = 1
        for entry in board_state:
            if i == 1:
                col1.append(entry)
                i += 1
            elif i == 2:
                col2.append(entry)
                i += 1
            elif i == 3:
                col3.append(entry)
                i += 1
            elif i == 4:
                col4.append(entry)
                i += 1
            elif i == 5:
                col5.append(entry)
                i += 1
            elif i == 6:
                col6.append(entry)
                i += 1
            elif i == 7:
                col7.append(entry)
                i = 1
        column_array = [col1, col2, col3, col4, col5, col6, col7]
        index = -1
        attemped_board_states = []
        winning_board_states = []
        tie_board_states = []
        board_state_to_return = []
        for col in column_array:
            '''
            for i, entry in enumerate(col):
                if entry == 0: 
                    if col[i + 1] != 0:
                        index = i
                        break
            '''
            for i in range(len(col) - 1, -1, -1):  # Start from the bottom row
                if col[i] == 0:
                    index = i
                    break

            if index != -1:
                col[index] = 2.0
                new_board_state = convert_to_board_state(col1, col2, col3, col4, col5, col6, col7)
                attemped_board_states.append(new_board_state)
                new_board_state_tensor = torch.tensor(new_board_state)
                winner = model(new_board_state_tensor)
                winner = winner.argmax(0)
                winner = int(winner.numpy())
                if winner == 2:
                    winning_board_states.append(new_board_state)
                elif winner == 0:
                    tie_board_states.append(new_board_state)
                col[index] = 0.0
                index = -1
        if not winning_board_states:
            if not tie_board_states:
                random_state = random.randrange(1, 8)
                board_state_to_return = attemped_board_states[random_state]
            else: 
                num_tie_boards = len(tie_board_states)
                if num_tie_boards == 1:
                    board_state_to_return = tie_board_states[0]
                else:
                    random_state = random.randrange(1, num_tie_boards)
                    board_state_to_return = tie_board_states[random_state]
        else:
            num_winning_boards = len(winning_board_states)
            if num_winning_boards == 1:
                board_state_to_return = winning_board_states[0]
            else:
                random_state = random.randrange(1, num_winning_boards)
                board_state_to_return = winning_board_states[random_state]

        return board_state_to_return


if __name__ == "__main__":
    input = [0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 2., 0., 0.,
            0., 1., 2., 2., 1., 1., 1.,
            0., 1., 2., 1., 2., 1., 2.,]
    difficulty = "hard"
    board_state = find_best_move(input, difficulty)
    print(board_state)

'''
            [0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0.,
            2., 0., 0., 0., 0., 0., 0.,]

            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 
            0.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0,
            0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]

'''
