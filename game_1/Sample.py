import math
import random

import numpy as np

import STcpClient

NUM_PLAYER = 4
DIRECTION = {1: (-1, -1), 2: (0, -1), 3: (1, -1), 4: (-1, 0), 6: (1, 0), 7: (-1, 1), 8: (0, 1), 9: (1, 1)}


class MinMaxNode:
    def __init__(self, turn, mapStat, sheepStat, heuristic="team", exploration="mean", teammate=None):
        self.turn = turn
        self.map = np.array(mapStat, dtype=int)
        self.sheep = np.array(sheepStat, dtype=int)
        self.heuristic = heuristic
        self.exploration = exploration
        self.teammate = teammate | {turn} if teammate else {turn}

    def IsTeamTurn(self):
        return self.turn in self.teammate

    def GetScore(self, heuristic=None):
        self.heuristic = heuristic if heuristic else self.heuristic

        if self.heuristic == "team":
            return self._GetTeamScore()
        if self.heuristic == "team-difference":
            return self._GetTeamScoreDifference()
        if self.heuristic == "team-winner-bonus":
            return self._GetTeamScoreWithWinnerBonus()
        if self.heuristic == "team-difference-winner-bonus":
            return self._GetTeamScoreDifferenceWithWinnerBonus()

        return -1

    def GetLegalStep(self, exploration=None):
        self.exploration = exploration if exploration else self.exploration

        if self.exploration == "mean":
            return self._GetMeanWeightLegalStep()
        if self.exploration == "three-random":
            return self._GetThreeRandomWeightLegalStep()
        if self.exploration == "mean-extreme":
            return self._GetMeanAndTwoExtremeWeightLegalStep()
        if self.exploration == "mean-random":
            return self._GetMeanAndTwoRandomWeightLegalStep()

        return ()

    def GetNextState(self, step):
        next = MinMaxNode((self.turn + 1) % NUM_PLAYER, self.map.copy(), self.sheep.copy(),
                          self.heuristic, self.exploration, self.teammate)

        pos, m, dir = step
        x, y = pos
        target_x, target_y = next._GetTargetCell(x, y, DIRECTION[dir][0], DIRECTION[dir][1])

        next.map[target_x][target_y] = self.turn
        next.sheep[x][y] -= m
        next.sheep[target_x][target_y] = m

        return next

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

    def _GetPlayerScore(self, player):
        return sum(self._GetArea(x, y, player)**1.25 for x in range(len(self.map)) for y in range(len(self.map)))

    def _GetArea(self, x, y, player):
        if x < 0 or x >= len(self.map) or y < 0 or y >= len(self.map) or self.map[x][y] != player:
            return 0

        self.map[x][y] = -1

        return 1 + sum(self._GetArea(x + dx, y + dy, player) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)))

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


def MinMax(playerID, mapStat, sheepStat, depth, heuristic, exploration) -> list | None:
    root = MinMaxNode(playerID, mapStat, sheepStat, heuristic, exploration)
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
    def AtBoundary(i, j):
        return i == 0 or i == len(mapStat) - 1 or j == 0 or j == len(mapStat) - 1 or \
            mapStat[i][j - 1] == -1 or mapStat[i][j + 1] == -1 or mapStat[i - 1][j] == -1 or mapStat[i + 1][j] == -1

    return [[i, j] for i in range(len(mapStat)) for j in range(len(mapStat)) if mapStat[i][j] == 0 and AtBoundary(i, j)]


'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)

    return: init_pos
    init_pos=[x,y],代表起始位置

'''


def InitPos(mapStat):
    # get legal positions
    legal_pos = GetLegalInitPos(mapStat)

    # choose randomly
    return random.choice(legal_pos)


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
    depth, heuristic, exploration = 25, "team", "mean-extreme"
    return MinMax(playerID, mapStat, sheepStat, depth, heuristic, exploration)


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
