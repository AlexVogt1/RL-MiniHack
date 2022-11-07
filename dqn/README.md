# MiniHack-Quest-Hard-v0 using REINFORCE

## Requirements

Please ensure that NLE (https://github.com/facebookresearch/nle) and MiniHack (https://github.com/facebookresearch/minihack) are installed. 

Pytorch is also required. Installation instructions can be found at: https://pytorch.org/get-started/locally/

All other requirements can be install using `python3 -m pip install numpy matplotlib gym tqdm`

## Training

The model can be trained in `train_dqn.py`. If you wish to train the dqn on a different environment change the `level_name` variable to the desired environment. Hyperparameter details can be found in the associated report. After training is complete, the saved models will be saved under `models/`. And the saved gifs are saved under `./gif`.

## Generating Videos

All the generated gifs are stored in `./gif`. All the models and videos are created and saved during training. Generating the gifs take up alot of compute so comment and uncomment those lines as you see fit.