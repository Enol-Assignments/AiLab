from typing import Tuple, Iterable

GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)


def IndexToMatrix(index):
    return divmod(index, 3)


def MatrixToIndex(row, column):
    return row * 3 + column


def ParseEightPuzzle(board):
    if isinstance(board, str):
        parts = board.split()
        arr = [int(x) for x in parts]
    elif isinstance(board, list) and len(board) == 3 and isinstance(board[0], list):
        arr = [x for row in board for x in row]
    else:
        arr = list(board)
    return tuple(arr)


def GetNeighbors(state: Tuple[int]) -> Iterable[Tuple[int]]:
    zero_idx = state.index(0)
    r, c = IndexToMatrix(zero_idx)
    moves = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            nidx = MatrixToIndex(nr, nc)
            lst = list(state)
            lst[zero_idx], lst[nidx] = lst[nidx], lst[zero_idx]
            moves.append(tuple(lst))
    return moves


def ManhattanDistance(state: Tuple[int], goal=GOAL_STATE) -> int:
    pos_goal = {val: i for i, val in enumerate(goal)}
    total = 0
    for i, val in enumerate(state):
        if val == 0:
            continue
        gi = pos_goal[val]
        r1, c1 = IndexToMatrix(i)
        r2, c2 = IndexToMatrix(gi)
        total += abs(r1 - r2) + abs(c1 - c2)
    return total
