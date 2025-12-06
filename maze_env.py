from maze_verifier import MazeVerifier
from prompts.maze_prompt import get_maze_prompt 
from base.env import Env
from typing import Optional
import random
from typing import List, Tuple
from collections import deque
from base.data import Data

class MazeEnv(Env):
  def __init__(self):
    super().__init__("Maze Shortest Path", MazeVerifier)
    
  def extract_answer(self, test_solution: str) -> str:
    """
    Extract the answer from the test solution
    @param test_solution: str
    @return: str
    """
    return self.verifier.extract_answer(test_solution)
  
  def generate(self, num_of_questions: int = 100, max_attempts: int = 100, difficulty: Optional[int] = 1):
    """
    Generate game questions and answers
    @param num_of_questions: int
		@param max_attempts: int
    @return: list of Data
    """
    dataset = []
    
    levels = {
      1: (5, 9), # easy
      2: (11, 15), # medium
      3: (17, 21) # hard
    }
    
    size_range = levels[difficulty]
    
    attempts = 0
    while len(dataset) < num_of_questions and attempts < max_attempts * num_of_questions:
      attempts += 1
      
      # Need odd sizes for generator
      h = (random.randint(size_range[0], size_range[1]) | 1)
      w = (random.randint(size_range[0], size_range[1]) | 1)
      
      grid, start, end = self._generate_maze_grid(h, w)
      gold_answer = self._solve_bfs(grid, start, end)
      if not gold_answer or len(gold_answer) < 5:
        continue
        
      maze_str = self._grid_to_text(grid, start, end)
      full_promt = get_maze_prompt(maze_str)
      
      data = Data(
        question=full_promt,
        answer=gold_answer,
        difficulty=difficulty,
        metadata={
          "grid": grid,
          "start": start,
          "end": end
        }
      )
      
      dataset.append(data)
        
    return dataset
      
        
  def _generate_maze_grid(self, height: int, width: int):
    grid = [[1 for _ in range(width)] for _ in range(height)]
    start_r, start_c = 1, 0
    grid[start_r][start_c] = 0
    
    stack = [(start_r, start_c)]
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)] # Moves through one cell
    
    while stack:
      r, c = stack[-1]
      random.shuffle(directions)
      moved = False
      
      for dr, dc in directions:
        nr, nc = r + dr, c + dc
        
        if not (1 <= nr < height - 1 and 1 <= nc < width - 1 and grid[nr][nc] == 1):
          continue
        
        grid[r + dr // 2][c + dc // 2] = 0
        grid[nr][nc] = 0
        stack.append((nr, nc))
        moved = True
        break
      
      if not moved:
        stack.pop()
        
    end_r, end_c = height - 2, width - 1
    
    grid[end_r][end_c] = 0
    grid[end_r][end_c - 1] = 0
    
    return grid, (start_r, start_c), (end_r, end_c)
  
  def _solve_bfs(self, grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> Optional[str]:
    rows = len(grid)
    cols = len(grid)
    queue = deque([start])
    visited = {start}
    history = [[-1] * cols for _ in range(rows)]
    
    moves = [(-1, 0, 'U'), (1, 0, 'D'), (0, 1, 'R'), (0, -1, 'L')]
    while queue:
      r, c = queue.popleft()
      
      if (r, c) == end:
        break
      
      for dr, dc, symbol in moves:
        nr, nc = r + dr, c + dc
        if (not (0 <= nr < rows and 0 <= nc < cols)) or grid[nr][nc] != 0 or ((nr, nc) in visited):
          continue
        visited.add((nr, nc))
        queue.append((nr, nc))
        history[nr][nc] = symbol
    
    if end not in visited:
      return None
    
    path = ""
    r, c = end
    while (r, c) != start:
      for dr, dc, symbol in moves:
        if history[r][c] == symbol:
          path += symbol
          r -= dr
          c -= dc
          break
    
    return path[::-1]
  
  def _grid_to_text(self, grid, start, end):
    """Grid to text convertion for promt"""
    lines = []
    for r in range(len(grid)):
      row_chars = []
      for c in range(len(grid[r])):
        if (r, c) == start:
          row_chars.append('S')
        elif (r, c) == end:
          row_chars.append('E')
        elif grid[r][c] == 1:
          row_chars.append('#')
        else:
          row_chars.append('.')
      lines.append("".join(row_chars))
    return "\n".join(lines)