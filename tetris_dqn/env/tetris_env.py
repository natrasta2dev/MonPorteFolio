# Classe TetrisEnv simulant l'environnement du jeu
# env/tetris_env.py

import numpy as np
import random
import pygame

pygame.display.set_mode((300, 600))
pygame.display.set_caption("Tetris AI Live")


class TetrisEnv:
    def __init__(self):
        self.width = 10
        self.height = 20
        self.board = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.pieces = {
            'I': [[1, 1, 1, 1]],
            'O': [[1, 1], [1, 1]],
            'T': [[0, 1, 0], [1, 1, 1]],
            'S': [[0, 1, 1], [1, 1, 0]],
            'Z': [[1, 1, 0], [0, 1, 1]],
            'J': [[1, 0, 0], [1, 1, 1]],
            'L': [[0, 0, 1], [1, 1, 1]],
        }
        self.reset()


    def reset(self):
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.score = 0
        self.game_over = False
        self.current_piece = self._new_piece()
        self.current_pos = [0, (self.width - len(self.current_piece[0])) // 2]
        return self._get_state()

    def _new_piece(self):
        key = random.choice(list(self.pieces.keys()))
        return [row[:] for row in self.pieces[key]]

    def step(self, action):
        """Action: 0 = gauche, 1 = droite, 2 = rotation, 3 = drop"""
        if self.game_over:
            return self._get_state(), 0, True, {}

        if action == 0:  # gauche
            self.current_pos[1] -= 1
            if self._collision():
                self.current_pos[1] += 1
        elif action == 1:  # droite
            self.current_pos[1] += 1
            if self._collision():
                self.current_pos[1] -= 1
        elif action == 2:  # rotation
            self.current_piece = self._rotate(self.current_piece)
            if self._collision():
                self.current_piece = self._rotate(self.current_piece, -1)
        elif action == 3:  # drop
            while not self._collision():
                self.current_pos[0] += 1
            self.current_pos[0] -= 1
            self._merge_piece()
            lines_cleared = self._clear_lines()
            reward = 5 * lines_cleared if lines_cleared > 0 else -2
            self.current_piece = self._new_piece()
            self.current_pos = [0, (self.width - len(self.current_piece[0])) // 2]
            if self._collision():
                self.game_over = True
                reward = -10
            return self._get_state(), reward, self.game_over, {}

        return self._get_state(), 0, self.game_over, {}

    def _collision(self):
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell == 0:
                    continue
                px = self.current_pos[1] + x
                py = self.current_pos[0] + y
                if px < 0 or px >= self.width or py >= self.height or self.board[py][px]:
                    return True
        return False

    def _merge_piece(self):
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    self.board[self.current_pos[0] + y][self.current_pos[1] + x] = cell

    def _clear_lines(self):
        lines_before = np.count_nonzero(np.all(self.board != 0, axis=1))
        self.board = np.array([row for row in self.board if not all(row)])  # remove full rows
        lines_after = self.height - self.board.shape[0]
        while self.board.shape[0] < self.height:
            self.board = np.vstack([np.zeros((1, self.width)), self.board])
        return lines_after

    def _rotate(self, matrix, k=1):
        return np.rot90(matrix, -k).tolist()

    def _get_state(self):
        flat_board = np.array(self.board).flatten()
        piece_x = self.piece_x if hasattr(self, 'piece_x') else 0
        piece_y = self.piece_y if hasattr(self, 'piece_y') else 0
        piece_type = list(self.pieces.keys()).index(self.current_piece_key) if hasattr(self, 'current_piece_key') else 0

        # Ajout de 3 éléments : piece_x, piece_y, type
        extras = np.array([piece_x, piece_y, piece_type])

        state = np.concatenate([flat_board, extras])

        # Pad si nécessaire pour atteindre exactement 208 éléments
        required_size = 208
        if len(state) < required_size:
            padding = np.zeros(required_size - len(state))
            state = np.concatenate([state, padding])
        elif len(state) > required_size:
            state = state[:required_size]

        return state.astype(np.float32)
