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

start_m = dataset[0]
test_m = dataset[3]

from maze_dataset.tokenization import MazeTokenizerModular, PromptSequencers
token_parsed_start_m = start_m.as_tokens(maze_tokenizer=MazeTokenizerModular(prompt_sequencer = PromptSequencers.AOTP()))

token_parsed_test_m = test_m.as_tokens(maze_tokenizer=MazeTokenizerModular(prompt_sequencer = PromptSequencers.AOTP()))

path_start_index = token_parsed_test_m.index("<PATH_START>")
token_parsed_test_m_no_sol = " ".join(token_parsed_test_m[:path_start_index+1])

start_mazes = " ".join(token_parsed_start_m) + "\n"

client = OpenAI(
    base_url="http://172.29.96.1:8080/v1", # "http://<Your api-server IP>:port"
    api_key = "sk-no-key-required"
)

#print(token_parsed_test_m_no_sol)
completion = client.chat.completions.create(
    model="LLaMA_CPP",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that generates solutions for mazes, going from an origin to a target"},
        {"role": "user", "content": f"Consider the following solved maze:\n{start_mazes}Solve the following maze:\n{token_parsed_test_m_no_sol}"},
    ]
 )
print(completion.choices[0].message.content)

print(start_mazes, token_parsed_test_m_no_sol)