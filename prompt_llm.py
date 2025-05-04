import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Make sure these are correct
API_KEY = os.getenv("API_KEY")
URL = None
MODEL = "A"

client = OpenAI(api_key=API_KEY, base_url=URL)

# If working with different maze sizes / generation methods, this will need to be changed
input_base_path = "mazes/gen_dfs_5x5"
output_base_path = "responses/gen_dfs_5x5"

for folder in os.listdir(input_base_path):

    folder_path = os.path.join(input_base_path, folder)
    if os.path.isdir(folder_path):

        output_folder_path = os.path.join(output_base_path, MODEL, folder)
        os.makedirs(output_folder_path, exist_ok=True)

        example_file_path = os.path.join(folder_path, "maze_example.txt")
        example_content = None
        if os.path.isfile(example_file_path):
            with open(example_file_path, "r") as example_file:
                example_maze = example_file.read()

        for file_name in [f"maze_{i}.txt" for i in range(1, 21)]:

            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, "r") as file:
                    maze = file.read()
                    
                    maze_input = ""
                    expected_output = ""

                    if maze.startswith("#"):
                        maze_input = maze.replace("X", " ")
                        expected_output = maze
                    else:
                        path_start_index = maze.find("<PATH_START>")
                        if path_start_index != -1:
                            maze_input = maze[:path_start_index]
                            expected_output = maze[path_start_index:]

                    completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": "You are a helpful assistant that generates solutions for mazes, going from an origin to a target."},
                              {"role": "user", "content": f"Consider the following solved maze configuration:\n{example_maze}. For the following maze, provide a path going from origin to target:\n{maze_input}"}],
                    )

                    response = completion.choices[0].message.content

                    output = "INPUT:\n" + maze_input + "\n\nEXPECTED SOLUTION:\n" + expected_output + "\n\nOUTPUT:\n" + response + "\n\nHUMAN VERIFICATION:"

                    output_file_path = os.path.join(output_folder_path, file_name)
                    with open(output_file_path, "w") as output_file:
                        output_file.write(output)