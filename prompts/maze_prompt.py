def get_maze_prompt(maze_map: str) -> str:
  """
  Generates a strict and clear prompt for the Maze Shortest Path task.
  
  @param maze_map: The ASCII representation of the maze.
  @return: Full prompt string.
  """
  return f"""Task: Maze Navigation

You are an intelligent agent placed in a 2D rectangular grid maze.
Your goal is to find the shortest path from 'S' (Start) to 'E' (End).

Map Legend:
- '#' : Wall (Cannot step here)
- '.' : Empty space (Walkable)
- 'S' : Start position
- 'E' : End position

Rules:
1. Valid moves: U (Up), D (Down), L (Left), R (Right). No diagonals.
2. Do not hit walls or go out of bounds.
3. Find the SHORTEST path from S to E (minimum number of moves).

INSTRUCTIONS FOR THINKING PROCESS:
1. First, identify the coordinates of 'S' and 'E'.
2. In the <think> tag, plan your path step-by-step.
3. For every move, write down the current coordinate and the next coordinate to ensure you don't hit a wall.
4. Double-check that your path actually reaches 'E'.

OUTPUT FORMAT: 
After your thinking process, provide the final move sequence inside <answer> tags.
Try faster find out the solution. 
Example:
<think>
[Plan the path concisely. Do not explain every single move with text. Use coordinate lists or short arrows.]
Example: Start (1,1) -> (1,2) -> (2,2)...
</think>
<answer>
[Type only path letters.]
Example: LLLLRLRLDDUUU...
</answer>

Here is the map of the maze:
{maze_map}"""