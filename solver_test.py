import os
import unittest

import numpy as np
from PIL import Image

from read_lock import solve_lock_dfs
from utils.img_utils import DEBUG_OUT_PATH


def gen_picks():
    ring_holes = []

    # generate 4 holes to poke out of the ring, they cannot be adjacent
    while len(ring_holes) < 4:
        possible_hole = np.random.randint(0, 32)
        valid = True
        if possible_hole in ring_holes:
            continue
        for hole in ring_holes:
            diff = abs(hole - possible_hole)
            if diff == 1 or (possible_hole == 0 and hole == 31) or (possible_hole == 31 and hole == 0):
                valid = False
        if valid:
            ring_holes.append(possible_hole)

    pick1 = np.full(32, False)
    pick2 = np.full(32, False)

    # pick 1-3 of the random holes to create a pick
    num_pick_holes = np.random.randint(1, 4)
    pick_holes = np.random.choice(ring_holes, num_pick_holes, replace=False)
    pick1[pick_holes] = True

    # the remaining holes go into the second pick
    remaining_holes = np.setdiff1d(ring_holes, pick_holes)
    pick2[remaining_holes] = True

    return pick1, pick2


def gen_puzzle(
    num_rings=2,
    total_picks=4,
):
    filled_puzzle = np.full((num_rings, 32), True)

    picks = []
    rolled_picks = []
    roll_vals = []

    for ring_num in range(num_rings):
        pick1, pick2 = gen_picks()
        this_ring_picks = [pick1, pick2]
        picks.append(this_ring_picks)

        # poke the holes and roll the picks
        for pick in this_ring_picks:
            filled_puzzle[ring_num][np.argwhere(pick == True)] = False
            roll = np.random.randint(0, 32)
            roll_vals.append(roll)
            rolled_picks.append(np.roll(pick, roll))

        assert np.count_nonzero(filled_puzzle[ring_num] & pick1 & pick2) == 0

    # fill the rest of the puzzle with duds
    pick_count = num_rings * 2
    while pick_count < total_picks:
        pick1, pick2 = gen_picks()
        if pick_count + 1 == total_picks:
            rolled_picks.append(pick1)
            pick_count += 1
        else:
            rolled_picks.append(pick1)
            rolled_picks.append(pick2)
            pick_count += 2

    return filled_puzzle, picks, rolled_picks, roll_vals


def draw_puzzle(puzzle: np.ndarray, picks: np.ndarray, rolled_picks: np.ndarray):
    """Draw the puzzle as a boolean image array."""

    puzzle = puzzle.transpose()

    base_puzzle_img = Image.fromarray(puzzle.astype(np.uint8) * 255)
    base_puzzle_img.save(os.path.join(DEBUG_OUT_PATH, "puzzle.bmp"))

    puzzle_w_picks = puzzle.copy()
    ring_len = puzzle.shape[1]
    # add a blank column for spacing
    puzzle_w_picks = np.concatenate((puzzle_w_picks, np.zeros((32, 1), dtype=np.uint8)), axis=1)
    for ring_num in range(ring_len):
        for pick_num, pick in enumerate(picks[ring_num]):
            puzzle_w_picks = np.concatenate((puzzle_w_picks, pick[:, np.newaxis]), axis=1)

    puzzle_w_picks = Image.fromarray(puzzle_w_picks.astype(np.uint8) * 255)
    puzzle_w_picks.save(os.path.join(DEBUG_OUT_PATH, "puzzle_w_picks.bmp"))


def test_puzzle(puzzle, rolled_picks, roll_vals):
    move_str = "%d[%d]"
    solve_str = ""
    # the last few picks are duds, so don't include them in the solve string
    solve_picks = rolled_picks[: len(roll_vals)]

    # shuffle the picks around and keep track of where they end up
    shuffled_indices = np.random.permutation(len(solve_picks))
    shuffled_picks = [solve_picks[i] for i in shuffled_indices]
    shuffled_rolls = [roll_vals[i] for i in shuffled_indices]

    for pick, roll_amount in enumerate(shuffled_rolls):
        undo_amount = 32 - roll_amount
        if undo_amount == 32:
            undo_amount = 0
        if undo_amount > 16:
            undo_amount = undo_amount - 32
        solve_str += move_str % (pick, undo_amount)
    solve_str += " - SOLVED"
    print("Solve string is:", solve_str)

    ret_val = solve_lock_dfs(puzzle, shuffled_picks, 0)
    print("DFS returned:", ret_val)
    assert solve_str == ret_val


class TestSolver(unittest.TestCase):
    """Various test cases for the solver."""

    def test_single_ring(self):
        """Basic test - single ring, 2 picks"""
        puzzle, picks, rolled_picks, roll_vals = gen_puzzle(1, 2)
        test_puzzle(puzzle, rolled_picks, roll_vals)

    def test_novice_diff(self):
        """Novice difficulty - two rings, 4 picks"""
        puzzle, picks, rolled_picks, roll_vals = gen_puzzle(2, 4)
        test_puzzle(puzzle, rolled_picks, roll_vals)

    def test_adv_diff(self):
        """Advanced difficulty - two rings, 6 picks (2 duds)"""
        puzzle, picks, rolled_picks, roll_vals = gen_puzzle(2, 6)
        test_puzzle(puzzle, rolled_picks, roll_vals)

    def test_expert_diff(self):
        """Expert difficulty - three rings, 9 picks (3 duds)"""
        puzzle, picks, rolled_picks, roll_vals = gen_puzzle(3, 9)
        test_puzzle(puzzle, rolled_picks, roll_vals)


if __name__ == "__main__":
    unittest.main()