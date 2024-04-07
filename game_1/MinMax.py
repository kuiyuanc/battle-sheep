import math
import random

import numpy as np

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
