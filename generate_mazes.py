
import os
from itertools import product

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
	name="dfs_5x5", # name is only for you to keep track of things
    seed=3, # trying to get a non-trivial example maze
	grid_n=5, # number of rows/columns in the lattice
	n_mazes=21, # number of mazes to generate
	maze_ctor=LatticeMazeGenerators.gen_dfs, # algorithm to generate the maze
    maze_ctor_kwargs=dict(do_forks=False), # additional parameters to pass to the maze generation algorithm
)

dataset: MazeDataset = MazeDataset.from_config(cfg)

maze_folder = "mazes" + os.path.sep + cfg.maze_ctor.__name__ + "_" + str(cfg.grid_n) + "x" + str(cfg.grid_n)
os.makedirs(maze_folder, exist_ok=True)

# Tokenization options
# coord_tokenizers = [CoordTokenizers.UT(), CoordTokenizers.CTT()]
# adj_list_tokenizers = [AdjListTokenizers.AdjListCoord(pre=False, post=True, shuffle_d0=True,edge_subset=EdgeSubsets.ConnectionEdges(walls=False), edge_permuter=EdgePermuters.RandomCoords()),
#                        AdjListTokenizers.AdjListCardinal(pre=False, post=True, shuffle_d0=True,edge_subset=EdgeSubsets.ConnectionEdges(walls=False), edge_permuter=EdgePermuters.RandomCoords()),]
# target_tokenizers = [TargetTokenizers.Unlabeled(post=False)]
# path_tokenizers = [PathTokenizers.StepSequence(step_size=StepSizes.Singles(), step_tokenizers=(StepTokenizers.Coord(),), pre=False, intra=False, post=False)]

# Create all possible combinations of tokenizers from above options
# tokenizer_list = [
#     MazeTokenizerModular(
#         prompt_sequencer=PromptSequencers.AOTP(
#             coord_tokenizer=coord,
#             adj_list_tokenizer=adj,
#             target_tokenizer=target,
#             path_tokenizer=path
#         )
#     )
#     for coord, adj, target, path in product(coord_tokenizers, adj_list_tokenizers, target_tokenizers, path_tokenizers)
# ]

tokenizer_list = [
    MazeTokenizerModular(
        prompt_sequencer=PromptSequencers.AOTP(
            coord_tokenizer=CoordTokenizers.UT(),
            adj_list_tokenizer=AdjListTokenizers.AdjListCoord(pre=False, post=True, shuffle_d0=True,edge_subset=EdgeSubsets.ConnectionEdges(walls=False), edge_permuter=EdgePermuters.RandomCoords()),
            target_tokenizer=TargetTokenizers.Unlabeled(post=False),
            path_tokenizer=PathTokenizers.StepSequence(step_size=StepSizes.Singles(), step_tokenizers=(StepTokenizers.Coord(),), pre=False, intra=False, post=False)
    )),
    MazeTokenizerModular(
        prompt_sequencer=PromptSequencers.AOTP(
            coord_tokenizer=CoordTokenizers.CTT(),
            adj_list_tokenizer=AdjListTokenizers.AdjListCoord(pre=False, post=True, shuffle_d0=True,edge_subset=EdgeSubsets.ConnectionEdges(walls=False), edge_permuter=EdgePermuters.RandomCoords()),
            target_tokenizer=TargetTokenizers.Unlabeled(post=False),
            path_tokenizer=PathTokenizers.StepSequence(step_size=StepSizes.Singles(), step_tokenizers=(StepTokenizers.Coord(),), pre=False, intra=False, post=False))),
    MazeTokenizerModular(
        prompt_sequencer=PromptSequencers.AOTP(
            coord_tokenizer=CoordTokenizers.UT(),
            adj_list_tokenizer=AdjListTokenizers.AdjListCardinal(pre=False, post=False, shuffle_d0=False,edge_subset=EdgeSubsets.ConnectionEdges(walls=False), edge_permuter=EdgePermuters.RandomCoords()),
            target_tokenizer=TargetTokenizers.Unlabeled(post=False),
            path_tokenizer=PathTokenizers.StepSequence(step_size=StepSizes.Singles(), step_tokenizers=(StepTokenizers.Cardinal(), StepTokenizers.Coord(),), pre=False, intra=False, post=False))),
    MazeTokenizerModular(
        prompt_sequencer=PromptSequencers.AOTP(
            coord_tokenizer=CoordTokenizers.CTT(),
            adj_list_tokenizer=AdjListTokenizers.AdjListCardinal(pre=False, post=False, shuffle_d0=False,edge_subset=EdgeSubsets.ConnectionEdges(walls=False), edge_permuter=EdgePermuters.RandomCoords()),
            target_tokenizer=TargetTokenizers.Unlabeled(post=False),
            path_tokenizer=PathTokenizers.StepSequence(step_size=StepSizes.Singles(), step_tokenizers=(StepTokenizers.Cardinal(), StepTokenizers.Coord(),), pre=False, intra=False, post=False)))]

# Save the ASCII representation of the mazes
ascii_folder = os.path.join(maze_folder, "ascii")
os.makedirs(ascii_folder, exist_ok=True)

maze_list = []
for maze in dataset.mazes:
    maze_list.append(maze.as_ascii())
for i, maze in enumerate(maze_list):
    if i == 0:
        maze_file = os.path.join(ascii_folder, "maze_example.txt")
    else:
        maze_file = os.path.join(ascii_folder, f"maze_{i}.txt")
    with open(maze_file, "w") as f:
        f.write(str(maze))

# Save the tokenized representation of the mazes for various tokenizers
for tokenizer in tokenizer_list:

    tokenizer_folder = os.path.join(maze_folder, tokenizer.prompt_sequencer.coord_tokenizer.__class__.__name__ + "_" + 
                                    tokenizer.prompt_sequencer.adj_list_tokenizer.__class__.__name__ + "_" + 
                                    tokenizer.prompt_sequencer.path_tokenizer.__class__.__name__)
    os.makedirs(tokenizer_folder, exist_ok=True)

    maze_list = []
    for maze in dataset.mazes:
        maze_list.append(maze.as_tokens(maze_tokenizer=tokenizer))

    for i, maze in enumerate(maze_list):
        if i == 0:
            maze_file = os.path.join(tokenizer_folder, "maze_example.txt")
        else:
            maze_file = os.path.join(tokenizer_folder, f"maze_{i}.txt")
        with open(maze_file, "w") as f:
            f.write(" ".join(maze))