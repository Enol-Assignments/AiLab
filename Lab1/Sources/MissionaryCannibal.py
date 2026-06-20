def IsValidState(state):
    missionary, cannibal, boat = state
    if not (0 <= missionary <= 3 and 0 <= cannibal <= 3 and boat in (0, 1)):
        return False
    left_m, left_c = missionary, cannibal
    right_m, right_c = 3 - missionary, 3 - cannibal
    if left_m > 0 and left_m < left_c:
        return False
    if right_m > 0 and right_m < right_c:
        return False
    return True


def GetNeighbors(state):
    missionary, cannibal, boat = state
    result = []
    moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]
    for dm, dc in moves:
        if boat == 1:
            nm = missionary - dm
            nc = cannibal - dc
            n_boat = 0
        else:
            nm = missionary + dm
            nc = cannibal + dc
            n_boat = 1
        ns = (nm, nc, n_boat)
        if IsValidState(ns):
            result.append(ns)
    return result


def Heuristic(state):
    missionary, cannibal, boat = state
    remaining = missionary + cannibal
    return (remaining + 1) // 2
