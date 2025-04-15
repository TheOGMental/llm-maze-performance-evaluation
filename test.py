#!/usr/bin/env python3
from openai import OpenAI
from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators

cfg: MazeDatasetConfig = MazeDatasetConfig(
	name="test", # name is only for you to keep track of things
	grid_n=5, # number of rows/columns in the lattice
	n_mazes=4, # number of mazes to generate
	maze_ctor=LatticeMazeGenerators.gen_dfs, # algorithm to generate the maze
    maze_ctor_kwargs=dict(do_forks=False), # additional parameters to pass to the maze generation algorithm
)

dataset: MazeDataset = MazeDataset.from_config(cfg)

m = dataset[0]

from maze_dataset.tokenization import MazeTokenizerModular, TokenizationMode
print(m.as_tokens(maze_tokenizer=MazeTokenizerModular(
    tokenization_mode=TokenizationMode.AOTP_UT_rasterized, max_grid_size=100,
)))

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1", # "http://<Your api-server IP>:port"
    api_key = "sk-no-key-required"
)
#completion = client.chat.completions.create(
#    model="LLaMA_CPP",
#    messages=[
#        {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."},
#        {"role": "user", "content": f"For the following maze, given a start point, a goal, and inaccessible areas, provide a solution from start to finish.\n {m.as_ascii().replace("X", " ")}"}
#    ]
#)
#print(completion.choices[0].message.content)
#print("To find the path from the origin point (S) to the target point (E) in the given maze, we can use a simple algorithm:\n\n1. Start at the origin point (S).\n2. Explore each of the four directions (up, down, left, and right) from the current position.\n3. If a path is found to the target point (E), return it. Otherwise, backtrack to the previous position and continue exploring the remaining directions.\n\nHere's the path from the origin point (S) to the target point (E) in the maze:\n\n1. S -> # -> # -> # -> # -> E\n\nSo, the path is S -> # -> # -> # -> # -> E.</s>")