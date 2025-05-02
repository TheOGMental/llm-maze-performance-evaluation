
import os

from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.tokenization import MazeTokenizerModular, PromptSequencers

cfg: MazeDatasetConfig = MazeDatasetConfig(
	name="test", # name is only for you to keep track of things
	grid_n=5, # number of rows/columns in the lattice
	n_mazes=30, # number of mazes to generate
	maze_ctor=LatticeMazeGenerators.gen_dfs, # algorithm to generate the maze
    maze_ctor_kwargs=dict(do_forks=False), # additional parameters to pass to the maze generation algorithm
)

dataset: MazeDataset = MazeDataset.from_config(cfg)

# Need to fill this out with various tokenization methods
tokenizer_list = [MazeTokenizerModular()]

for tokenizer in tokenizer_list:
    mazes = []
    for maze in dataset.mazes:
        maze = maze.as_tokens(maze_tokenizer=tokenizer)
        mazes.append(maze)

    maze_folder = "mazes"
    os.makedirs(maze_folder, exist_ok=True)
    tokenizer_folder = os.path.join(maze_folder, tokenizer.__class__.__name__) # Name untested
    os.makedirs(tokenizer_folder, exist_ok=True)

    for i, maze in enumerate(mazes):
        maze_file = os.path.join(tokenizer_folder, f"maze_{i + 1}.txt")
        with open(maze_file, "w") as f:
            f.write(maze)