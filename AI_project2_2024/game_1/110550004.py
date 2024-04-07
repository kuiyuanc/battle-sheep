import STcpClient
import numpy as np
import random
import math

class GameState:
    def __init__(self, board, sheepStat, player_turn, player_ID):
        self.board = np.array(board)  # Board state, including obstacles and territories
        self.sheepStat = np.array(sheepStat)  # Distribution of sheep across the board
        self.player_turn = player_turn  # The player whose turn is currently
        self.player_ID = player_turn
    
    def copy(self):
        # Return a deep copy of the game state
        return GameState(self.board.copy(), self.sheepStat.copy(), self.player_turn, self.player_ID)
    
    def get_valid_moves(self):
        valid_moves = []
        for x in range(12):
            for y in range(12):
                if self.sheepStat[x][y] > 1 and self.board[x][y] == self.player_turn:
                    for direction in range(1, 10):
                        if direction == 5:
                            continue

                        step_count = self.is_valid_direction(x, y, direction)
                        if step_count > 0:
                            for split_group_count in range(1, self.sheepStat[x][y]):
                                valid_moves.append(((x, y), split_group_count, direction, step_count))
        return valid_moves
    
    def is_valid_direction(self, x, y, direction):
        dx, dy = {1: (-1, -1), 2: (-1, 0), 3: (-1, 1),
            4: ( 0, -1), 5: ( 0, 0), 6: ( 0, 1),
            7: ( 1, -1), 8: ( 1, 0), 9: ( 1, 1)}[direction]
        nx, ny = x + dx, y + dy

        # First step must be within bounds and unoccupied
        if not (0 <= nx < 12 and 0 <= ny < 12 and self.board[nx][ny] == 0 and self.sheepStat[nx][ny] == 0):
            return 0
        
        count = 0

        # Further steps: check until hitting an obstacle, another sheep, or the edge
        while 0 <= nx < 12 and 0 <= ny < 12:
            count += 1
            nx += dx
            ny += dy
            if not (0 <= nx < 12 and 0 <= ny < 12):  # Stop if next step goes out of bounds
                break
            if self.board[nx][ny] != 0 or self.sheepStat[nx][ny] > 0:  # Stop if next step hits an obstacle or sheep
                break

        return count  # The direction is valid if the sheep can start moving, return the steps count

    def make_move(self, move):
        (x, y), split_group_count, direction, step_count = move
        #if not self.is_valid_direction(x, y, direction):
        #    return False  # If the move isn't valid, return False
        
        # Calculate the new position for the split group of sheep
        new_x, new_y = self.calculate_new_position(x, y, direction, step_count)
        
        # Update sheepStat for the new and original positions
        self.sheepStat[new_x][new_y] = split_group_count
        self.sheepStat[x][y] -= split_group_count
        
        return True  # Return True to indicate the move was successfully made
    
    def calculate_new_position(self, x, y, direction, step_count):
        dx, dy = {1: (-1, -1), 2: (-1, 0), 3: (-1, 1),
                  4: ( 0, -1), 5: ( 0, 0), 6: ( 0, 1),
                  7: ( 1, -1), 8: ( 1, 0), 9: ( 1, 1)}[direction]
        
        nx, ny = x, y
        # Keep moving in the direction until an obstacle is reached or edge of board
        nx += dx * step_count
        ny += dy * step_count
        
        return nx, ny
    
    def next_player_turn(self):
        # Assuming player IDs are 1 through 4 and stored in self.player_turn
        self.player_turn = self.player_turn % 4 + 1  # Rotate player_turn to the next player

    def evaluate_winner(self):
        player_scores = {player_id: 0 for player_id in range(1, 5)}  # Assuming 4 players
        visited = set()

        def dfs(board, x, y, player_id, visited):
            if (x, y) in visited or not (0 <= x < 12 and 0 <= y < 12):
                return 0
            if board[x][y] != player_id:
                return 0
            visited.add((x, y))
            return 1 + sum(dfs(board, x + dx, y + dy, player_id, visited) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)])
        
        for x in range(12):
            for y in range(12):
                player_id = self.board[x][y]
                if player_id in player_scores and (x, y) not in visited:
                    region_size = dfs(self.board, x, y, player_id, visited)
                    # Add the score of this region to the player's total score
                    player_scores[player_id] += (region_size ** 1.25)
        
        # Round scores and determine the winner
        for player_id in player_scores:
            player_scores[player_id] = round(player_scores[player_id])
        
        winner = max(player_scores, key=player_scores.get)
        return winner, player_scores  # Returns the winner and the scores of all players




class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
    
    def select(self):
        best_score = float('-inf')
        best_child = None
        for child in self.children:
            score = child.wins / child.visits + math.sqrt(2 * math.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self):
        valid_moves = self.game_state.get_valid_moves()
        for move in valid_moves:
            new_game_state = self.game_state.copy()
            new_game_state.make_move(move)
            new_node = MCTSNode(new_game_state, self, move)
            self.children.append(new_node)

    def simulate(self):
        simulated_state = self.game_state.copy()
        while not simulated_state.game_end():
            valid_moves = simulated_state.get_valid_moves()
            if not valid_moves:  # If no valid moves, skip to the next player's turn
                simulated_state.next_player_turn()
                continue
            move = random.choice(valid_moves)  # Randomly select a move for the current player
            simulated_state.make_move(move)
            simulated_state.next_player_turn()  # Move to the next player's turn
            
        return simulated_state.evaluate_winner()


    def backpropagate(self, winner):
        self.visits += 1
        if winner == self.game_state.player_ID:
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(winner)

    def game_end(self):
        return not self.game_state.get_valid_moves()
    

    

# Identofy the edge position of the board, for the initial position
def find_edge_pos(board):
    edge_pos = []
    rows, cols = board.shape
    for x in range(rows):
        for y in range(cols):
            if board[x][y] == 0: # The chosen position must be empty
                if x == 0 or x == rows - 1 or y == 0 or y == cols - 1 or is_adjacent_to_obstacle(board, x, y, rows, cols): # Check if it's on the boundary or orthogonally adjacent to an obstacle
                    edge_pos.append([x, y])
    return edge_pos

# Used in find_edge_pos, to check if a position is orthogonally adjacent to an obstacle
def is_adjacent_to_obstacle(board, x, y, rows, cols):
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and board[nx][ny] == -1:
            return True
    return False

def evaluate_open_area(board, pos):
    visited = set()
    def dfs(x, y):
        if (x, y) in visited or not (0 <= x < 12 and 0 <= y < 12) or board[x][y] != 0:
            return 0
        visited.add((x, y))
        return 1 + dfs(x+1, y) + dfs(x-1, y) + dfs(x, y+1) + dfs(x, y-1)
    
    return dfs(pos[0], pos[1])



def run_MCTS(root_state, iterations):
    root_node = MCTSNode(root_state)

    # MCTS selection
    for _ in range(iterations):
        node = root_node
        while node.children and not node.game_end():
            node = node.select()

        # MCTS expansion
        if not node.game_end():
            node.expand()
            node = random.choice(node.children)

        # MCTS simulation
        winner = node.simulate()

        # MCTS backpropagation
        node.backpropagate(winner)

    return max(root_node.children, key=lambda x: x.wins / x.visits).move

'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=[x,y],代表起始位置
    
'''


def InitPos(mapStat):
    board = np.array(mapStat)
    edge_pos = find_edge_pos(board)

    best_pos = None
    best_open_area = -1

    for pos in edge_pos:
        open_area = evaluate_open_area(board, pos)

        if open_area > best_open_area:
            best_pos = pos
            best_open_area = open_area

    return best_pos if best_pos else [0, 0]


'''
    產出指令
    
    input: 
    playerID: 你在此局遊戲中的角色(1~4)
    mapStat : 棋盤狀態(list of list), 為 12*12矩陣, 
              0=可移動區域, -1=障礙, 1~4為玩家1~4佔領區域
    sheepStat : 羊群分布狀態, 範圍在0~16, 為 12*12矩陣

    return Step
    Step : 3 elements, [(x,y), m, dir]
            x, y 表示要進行動作的座標 
            m = 要切割成第二群的羊群數量
            dir = 移動方向(1~9),對應方向如下圖所示
            1 2 3
            4 X 6
            7 8 9
'''
def GetStep(playerID, mapStat, sheepStat):
    current_state = GameState(mapStat, sheepStat, playerID)
    best_move = run_MCTS(current_state, 100)
    print("Current state")
    return best_move


# player initial
(id_package, playerID, mapStat) = STcpClient.GetMap()
init_pos = InitPos(mapStat)
STcpClient.SendInitPos(id_package, init_pos)

# start game
while (True):
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)

    STcpClient.SendStep(id_package, Step)
