import math
import random

import numpy as np

import STcpClient

NUM_PLAYER = 4
DIRECTION = {1: (-1, -1), 2: (0, -1), 3: (1, -1), 4: (-1, 0), 6: (1, 0), 7: (-1, 1), 8: (0, 1), 9: (1, 1)}
MCTS_UPPERBOUND = 20


class MinMaxNode:
    def __init__(self, turn, mapStat, sheepStat, heuristic="team", strategy="mean", upperbound=MCTS_UPPERBOUND, teammate=None):
        self.turn = turn
        self.map = np.array(mapStat, dtype=int)
        self.sheep = np.array(sheepStat, dtype=int)
        self.heuristic = heuristic
        self.heuristics = {
            "team": self._GetTeamScore,
            "team-difference": self._GetTeamScoreDifference,
            "team-winner-bonus": self._GetTeamScoreWithWinnerBonus,
            "team-difference-winner-bonus": self._GetTeamScoreDifferenceWithWinnerBonus
        }
        self.strategy = strategy
        self.strategies = {
            "mean": self._GetMeanWeightLegalStep,
            "three-random": self._GetThreeRandomWeightLegalStep,
            "mean-extreme": self._GetMeanAndTwoExtremeWeightLegalStep,
            "mean-random": self._GetMeanAndTwoRandomWeightLegalStep,
            "mcts": self._GetRandomLegalStep
        }
        self.upperbound = upperbound
        self.teammate = teammate | {turn} if teammate else {turn}

    def IsTeamTurn(self):
        return self.turn in self.teammate

    def GetScore(self):
        '''
        Get the minmax score of current state based on the selected heuristic function
         - Select the heuristic function in GetStep
        '''
        return self.heuristics[self.heuristic]()

    def GetLegalStep(self):
        '''
        Get the legal step of current state based on the selected strategy function
         - Select the strategy function in GetStep
         - Set the fraction in GetStep
        '''
        return self.strategies[self.strategy]()

    def GetNextState(self, step):
        next = MinMaxNode((self.turn + 1) % NUM_PLAYER, self.map.copy(), self.sheep.copy(),
                          self.heuristic, self.strategy, self.upperbound, self.teammate)

        pos, m, dir = step
        x, y = pos
        target_x, target_y = next._GetTargetCell(x, y, DIRECTION[dir][0], DIRECTION[dir][1])

        next.map[target_x][target_y] = self.turn
        next.sheep[x][y] -= m
        next.sheep[target_x][target_y] = m

        return next

    '''

    heuristic functions

    '''

    def _GetTeamScore(self):
        return sum(self._GetPlayerScore(player) for player in self.teammate)

    def _GetTeamScoreDifference(self):
        return self._GetTeamScore() - sum(self._GetPlayerScore(player) for player in set(range(1, NUM_PLAYER + 1)) - self.teammate)

    def _GetTeamScoreWithWinnerBonus(self, *, grouped=False):
        score = [self._GetPlayerScore(player) for player in range(1, NUM_PLAYER + 1)]
        team_score = sum(score[player] for player in self.teammate)
        is_winner = sum(score) - team_score < team_score if grouped else team_score == max(score)
        return team_score + (100 if is_winner else 0)

    def _GetTeamScoreDifferenceWithWinnerBonus(self, *, grouped=False):
        score = [self._GetPlayerScore(player) for player in range(1, NUM_PLAYER + 1)]
        team_score = sum(score[player] for player in self.teammate)
        opponent_score = sum(score) - team_score
        is_winner = opponent_score < team_score if grouped else team_score == max(score)
        return team_score + (100 if is_winner else 0) - opponent_score

    '''

    Called by heuristic

    '''

    def _GetPlayerScore(self, player):
        return sum(self._GetArea(x, y, player)**1.25 for x in range(len(self.map)) for y in range(len(self.map)))

    def _GetArea(self, x, y, player):
        if x < 0 or x >= len(self.map) or y < 0 or y >= len(self.map) or self.map[x][y] != player:
            return 0

        self.map[x][y] = -1

        return 1 + sum(self._GetArea(x + dx, y + dy, player) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)))

    '''

    strategy functions

    '''

    def _GetMeanWeightLegalStep(self):
        return tuple([(x, y), self.sheep[x][y] // 2, dir] for x, y, dir in self._GetLegalPosAndDir())

    def _GetThreeRandomWeightLegalStep(self):
        return tuple([(x, y), m, dir] for x, y, dir in self._GetLegalPosAndDir()
                     for m in random.sample(range(1, self.sheep[x][y]), min(3, self.sheep[x][y] - 1)))

    def _GetMeanAndTwoExtremeWeightLegalStep(self):
        return tuple([(x, y), m, dir] for x, y, dir in self._GetLegalPosAndDir()
                     for m in {1, self.sheep[x][y] // 2, self.sheep[x][y] - 1})

    def _GetMeanAndTwoRandomWeightLegalStep(self):
        return tuple([(x, y), m, dir] for x, y, dir in self._GetLegalPosAndDir()
                     for m in (range(1, self.sheep[x][y]) if self.sheep[x][y] <= 4 else
                               {random.randint(1, math.floor(self.sheep[x][y] / 2) - (self.sheep[x][y] % 2 == 0)),
                                self.sheep[x][y] // 2,
                                random.randint(math.ceil(self.sheep[x][y] / 2) + (self.sheep[x][y] % 2 == 0), self.sheep[x][y] - 1)}))

    def _GetRandomLegalStep(self):
        legal_step = tuple([(x, y), m, dir] for x, y, dir in self._GetLegalPosAndDir() for m in range(1, self.sheep[x][y]))
        return tuple(random.sample(legal_step, self.upperbound)) if self.upperbound < len(legal_step) else legal_step

    '''

    called by strategy

    '''

    def _GetLegalPosAndDir(self):
        return tuple([x, y, dir] for x in range(len(self.map)) for y in range(len(self.map))
                     if self.map[x][y] == self.turn and self.sheep[x][y] > 1
                     for dir, d in DIRECTION.items()
                     if 0 <= x + d[0] < len(self.map) and 0 <= y + d[1] < len(self.map) and self.map[x + d[0]][y + d[1]] == 0)

    def _GetTargetCell(self, x, y, dx, dy):
        while 0 <= x + dx < len(self.map) and 0 <= y + dy < len(self.map) and self.map[x + dx][y + dy] == 0:
            x, y = x + dx, y + dy
        return x, y


def MinMaxSearch(node: MinMaxNode, depth: int, alpha=-np.inf, beta=np.inf) -> float:
    legal_step = node.GetLegalStep()

    if depth == 0 or not legal_step:
        return node.GetScore()

    if node.IsTeamTurn():
        score = -np.inf
        for step in legal_step:
            score = max(score, MinMaxSearch(node.GetNextState(step), depth - 1, alpha, beta))
            if score > beta:
                break
            alpha = max(alpha, score)
    else:
        score = np.inf
        for step in legal_step:
            score = min(score, MinMaxSearch(node.GetNextState(step), depth - 1, alpha, beta))
            if score < alpha:
                break
            beta = min(beta, score)

    return score


def MinMax(playerID, mapStat, sheepStat, depth, heuristic, strategy, upperbound, teammate) -> list | None:
    root = MinMaxNode(playerID, mapStat, sheepStat, heuristic, strategy, upperbound, teammate)
    legal_step = root.GetLegalStep()

    if depth == 0 or not legal_step:
        return

    alpha, beta, score, selected_step = -np.inf, np.inf, -np.inf, [(0, 0), 0, 1]
    for step in legal_step:
        next_score = MinMaxSearch(root.GetNextState(step), depth * NUM_PLAYER - 1, alpha, beta)
        if score <= next_score:
            selected_step = step if score < next_score else random.choice((step, selected_step))
        score = max(score, next_score)
        if score > beta:
            break
        alpha = max(alpha, score)

    return selected_step


def GetLegalInitPos(mapStat):
    def AtBoundary(x, y):
        return x == 0 or x == len(mapStat) - 1 or y == 0 or y == len(mapStat) - 1 or \
            mapStat[x][y - 1] == -1 or mapStat[x][y + 1] == -1 or mapStat[x - 1][y] == -1 or mapStat[x + 1][y] == -1

    return [[x, y] for x in range(len(mapStat)) for y in range(len(mapStat)) if mapStat[x][y] == 0 and AtBoundary(x, y)]


def GetRandomInitPos(mapStat):
    return random.choice(GetLegalInitPos(mapStat))


def GetWidestInitPos(mapStat):
    legal_pos = GetLegalInitPos(mapStat)
    scores = [sum(GetDistanceToBoundary(mapStat, x, y, dx, dy) for dx, dy in DIRECTION.values()) for x, y in legal_pos]
    return random.choice([pos for score, pos in zip(scores, legal_pos) if score == max(scores)])


def GetMeanInitPos(mapStat):
    legal_pos = GetLegalInitPos(mapStat)
    scores = []
    for x, y in legal_pos:
        distance = [GetDistanceToBoundary(mapStat, x, y, dx, dy) for dx, dy in DIRECTION.values()]
        short = len([i for i in distance if i <= 3])
        scores.append((4 - short) * (short - 4) * 10 + (0 if short == 8 else sum(i for i in distance if i > 3) / (8 - short)))
    return random.choice([pos for score, pos in zip(scores, legal_pos) if score == max(scores)])


def GetDistanceToBoundary(mapStat, x, y, dx, dy):
    distance = 0
    while 0 <= x + dx < len(mapStat) and 0 <= y + dy < len(mapStat) and mapStat[x + dx][y + dy] == 0:
        x, y, distance = x + dx, y + dy, distance + 1
    return distance


def find_edge_pos(board):
    edge_pos = []
    rows, cols = board.shape
    for x in range(cols):
        for y in range(rows):
            if board[x][y] == 0:  # The chosen position must be empty
                if x == 0 or x == rows - 1 or y == 0 or y == cols - 1 or is_adjacent_to_obstacle(board, x, y, rows, cols):  # Check if it's on the boundary or orthogonally adjacent to an obstacle
                    edge_pos.append([x, y])
    return edge_pos


def is_adjacent_to_obstacle(board, x, y, rows, cols):
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < cols and 0 <= ny < rows and board[nx][ny] == -1:
            return True
    return False


def evaluate_position_reachability(board, pos):
    x, y = pos[0], pos[1]
    evaluation = 1
    for direction in range(1, 9 + 1):
        if direction == 5:
            continue

        dx, dy = {1: (-1, -1), 2: (0, -1), 3: (1, -1),
                  4: (-1, 0),             6: (1, 0),
                  7: (-1, 1), 8: (0, 1), 9: (1, 1)}[direction]
        nx, ny = x + dx, y + dy
        # print(f"direction: {direction}")
        count = 1
        # First step must be within bounds and unoccupied
        if not (0 <= nx < 12 and 0 <= ny < 12 and board[nx][ny] == 0):
            continue

        # Further steps: check until hitting an obstacle, another sheep, or the edge
        while 0 <= nx < 12 and 0 <= ny < 12:
            count += 1
            nx += dx
            ny += dy
            if not (0 <= nx < 12 and 0 <= ny < 12):  # Stop if next step goes out of bounds
                break
            if board[nx][ny] != 0:  # Stop if next step hits an obstacle or sheep
                break

        evaluation *= count

    standard = 1 * (4**4) * (7**3)
    point = 0

    if evaluation < standard:
        point = evaluation / standard
    else:
        point = standard / evaluation

    print(f"{pos}: {point: .5f}")
    return point


def GetReachability(mapStat):
    print("mapStat: \n", mapStat)
    print("Initialized position")
    board = np.array(mapStat)
    edge_pos = find_edge_pos(board)

    best_pos = None
    best_point = -1
    print(len(edge_pos))
    for pos in edge_pos:
        point = evaluate_position_reachability(board, pos)

        if point > best_point:
            best_pos = pos
            best_point = point

    print("")
    return best_pos if best_pos else [0, 0]


'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)

    return: init_pos
    init_pos=[x,y],代表起始位置

'''


def InitPos(mapStat):
    '''
    tuning parameters
        - strategy: {"random", "widest", "mean", "reachability"}
    '''
    strategies = {
        "random": GetRandomInitPos,
        "widest": GetWidestInitPos,
        "mean": GetMeanInitPos,
        "reachability": GetReachability
    }
    strategy = "reachability"
    return strategies[strategy](mapStat)


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
    '''
    tuning parameters
        - depth: depth of searching tree = 'depth' * NUM_PLAYERS
        - heuristic: {"team", "team-difference", "team-winner-bonus", "team-winner-bonus-difference"}
        - strategy: {"mean", "three-ramdom", "mean-extreme", "mean-random", "mcts"}
        - upperbound: the upper bound of how many legal steps are sampled in a layer of MCTS searching tree
    '''
    depth, heuristic, strategy, upperbound, teammate = 15, "team-winner-bonus", "mcts", 20, None
    return MinMax(playerID, mapStat, sheepStat, depth, heuristic, strategy, upperbound, teammate)


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