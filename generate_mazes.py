
import os

from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.tokenization import (AdjListTokenizers,
                                       CoordTokenizers, 
                                       EdgePermuters, 
                                       EdgeSubsets, 
                                       MazeTokenizer, 
                                       MazeTokenizerModular, 
                                       PathTokenizers, 
                                       PromptSequencers, 
                                       StepSizes, 
                                       StepTokenizers, 
                                       TargetTokenizers, 
                                       TokenizationMode, 
                                       _TokenizerElement,)

cfg: MazeDatasetConfig = MazeDatasetConfig(
	name="test", # name is only for you to keep track of things
	grid_n=5, # number of rows/columns in the lattice
	n_mazes=31, # number of mazes to generate
	maze_ctor=LatticeMazeGenerators.gen_dfs, # algorithm to generate the maze
    maze_ctor_kwargs=dict(do_forks=False), # additional parameters to pass to the maze generation algorithm
)

dataset: MazeDataset = MazeDataset.from_config(cfg)

maze_folder = "mazes"
os.makedirs(maze_folder, exist_ok=True)

tokenizer_list = [MazeTokenizerModular(prompt_sequencer = PromptSequencers.AOTP(coord_tokenizer=CoordTokenizers.UT(),
                                                                                adj_list_tokenizer=AdjListTokenizers.AdjListCoord(pre=False, post=True, shuffle_d0=True,edge_subset=EdgeSubsets.ConnectionEdges(walls=False), edge_permuter=EdgePermuters.RandomCoords()),
                                                                                target_tokenizer=TargetTokenizers.Unlabeled(post=False),
                                                                                path_tokenizer=PathTokenizers.StepSequence(step_size=StepSizes.Singles(), step_tokenizers=(StepTokenizers.Coord(),), pre=False, intra=False, post=False))),
                  MazeTokenizerModular(prompt_sequencer = PromptSequencers.AOTP(coord_tokenizer=CoordTokenizers.CTT(),
                                                                                adj_list_tokenizer=AdjListTokenizers.AdjListCoord(pre=False, post=True, shuffle_d0=True,edge_subset=EdgeSubsets.ConnectionEdges(walls=False), edge_permuter=EdgePermuters.RandomCoords()),
                                                                                target_tokenizer=TargetTokenizers.Unlabeled(post=False),
                                                                                path_tokenizer=PathTokenizers.StepSequence(step_size=StepSizes.Singles(), step_tokenizers=(StepTokenizers.Coord(),), pre=False, intra=False, post=False)))
                 ]

for tokenizer in tokenizer_list:
    maze_list = []
    for maze in dataset.mazes:
        maze_list.append(maze.as_tokens(maze_tokenizer=tokenizer))

    tokenizer_folder = os.path.join(maze_folder, tokenizer.prompt_sequencer.coord_tokenizer.__class__.__name__ + "_" + tokenizer.prompt_sequencer.adj_list_tokenizer.__class__.__name__ + "_" + tokenizer.prompt_sequencer.path_tokenizer.__class__.__name__)
    os.makedirs(tokenizer_folder, exist_ok=True)

    for i, maze in enumerate(maze_list):
        if i == 0:
            maze_file = os.path.join(tokenizer_folder, "maze_example.txt")
        else:
            maze_file = os.path.join(tokenizer_folder, f"maze_{i}.txt")

        with open(maze_file, "w") as f:
            f.write(" ".join(maze))