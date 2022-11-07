# MiniHack-Quest-Hard-v0 using REINFORCE

## Requirements

Please ensure that NLE (https://github.com/facebookresearch/nle) and MiniHack (https://github.com/facebookresearch/minihack) are installed. 

Pytorch is also required. Installation instructions can be found at: https://pytorch.org/get-started/locally/

All other requirements can be install using `python3 -m pip install numpy matplotlib gym tqdm`

## Training

The model can be trained in `reinforce_train.py`. Hyperparameter details can be found in the associated report. After training is complete, the final saved model will be saved uner `model/`

## Testing 

Testing in the MiniHack-Quest-Hard-v0 enviroment is located in `reinforce_test.py`. Each subgoal can be found in corresponding file as described below:

|     Sub-Goal     |          File            |
|:----------------:|:------------------------:|
|       Maze       | `reinforce_test_maze.py` |
|     Lava Room    | `reinforce_test_lava.py` |
| Finding the Exit | `reinforce_test_room.py` |

## Generating Videos

The `reinforce_video.py` file runs the trained model stored in `./model` in the  MiniHack-Quest-Hard-v0 enviroemnt and generates a gif in `./gifs`