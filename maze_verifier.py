import re
from base.verifier import Verifier
from base.data import Data

class MazeVerifier(Verifier):
  def __init__(self):
    super().__init__()
        
  def extract_answer(self, test_solution: str) -> str:
    """
    Extract the answer from the test solution
    Find largest subsequence of symbols U, D, L, R
    @param test_solution: str
    @return: str
    """
    matches = re.findall(r'[UDLR]+', test_solution.upper())
    if not matches:
      return ""
    return max(matches, key=len)

  def verify(self, data: Data, test_answer: str) -> bool:
    """
    Verify whether the test answer is consistent with the gold answer
    @param data: Data
    @param test_answer: str
    @return: bool
    """
    test_answer = self.extract_answer(test_answer)
    
    grid = data.metadata.get('grid')
    start = data.metadata.get('start') 
    end = data.metadata.get('end')
    gold_answer = data.answer
    
    if len(test_answer) != len(gold_answer):
      return False
    
    rows = len(grid)
    cols = len(grid[0])
    
    cur_r, cur_c = start
    
    moves = {
      'U': (-1, 0),
      'D': (1, 0),
      'L': (0, -1),
      'R': (0, 1)
    }
    
    for move_symbol in test_answer:
      if move_symbol not in moves:
        return False
      
      dr, dc = moves[move_symbol]
      next_r, next_c = cur_r + dr, cur_c + dc
      
      if not (0 <= next_r < rows and 0 <= next_c < cols):
        return False
      
      if grid[next_r][next_c] == 1:
        return False
      
      cur_r, cur_c = next_r, next_c
    
    if (cur_r, cur_c) != tuple(end):
      return False
    
    return True
    
    