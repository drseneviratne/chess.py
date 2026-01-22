import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pygame
import sys
import math
import numpy as np

MODEL_PATH = "hand_landmarker.task"
WIDTH, HEIGHT = 800, 800
BOARD_SIZE = 8
SQUARE_SIZE = 80
BOARD_OFFSET_X = (WIDTH - BOARD_SIZE * SQUARE_SIZE) // 2
BOARD_OFFSET_Y = (HEIGHT - BOARD_SIZE * SQUARE_SIZE) // 2
HOVER_TIME = 45  # frames to hover (0.75 seconds at 60fps)

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_chess_piece(screen, piece_type, color, x, y, size):
    """Draw chess piece using geometric primitives at center position (x, y)"""

    # Color setup
    piece_color = (255, 255, 255) if color == 'white' else (30, 30, 30)
    outline_color = (30, 30, 30) if color == 'white' else (255, 255, 255)

    base_y = y + size // 3

    if piece_type == 'pawn':
        # Circle head + small base
        pygame.draw.circle(screen, piece_color, (x, y - size//6), size//4)
        pygame.draw.circle(screen, outline_color, (x, y - size//6), size//4, 2)
        pygame.draw.rect(screen, piece_color, (x - size//5, base_y - size//8, size//2.5, size//5))
        pygame.draw.rect(screen, outline_color, (x - size//5, base_y - size//8, size//2.5, size//5), 2)

    elif piece_type == 'rook':
        # Castle tower with crenellations
        tower_width = size // 2
        pygame.draw.rect(screen, piece_color, (x - tower_width//2, y - size//4, tower_width, size//2))
        pygame.draw.rect(screen, outline_color, (x - tower_width//2, y - size//4, tower_width, size//2), 2)
        # Crenellations (notches at top)
        for i in range(3):
            notch_x = x - tower_width//2 + (i * tower_width//3) + tower_width//6
            pygame.draw.rect(screen, piece_color, (notch_x - size//16, y - size//3, size//8, size//8))

    elif piece_type == 'knight':
        # L-shaped horse head silhouette
        points = [
            (x - size//6, base_y),
            (x - size//6, y - size//8),
            (x, y - size//3),
            (x + size//5, y - size//6),
            (x + size//6, base_y)
        ]
        pygame.draw.polygon(screen, piece_color, points)
        pygame.draw.polygon(screen, outline_color, points, 2)
        # Eye dot
        pygame.draw.circle(screen, outline_color, (x + size//12, y - size//5), 2)

    elif piece_type == 'bishop':
        # Circle top + diagonal body
        pygame.draw.circle(screen, piece_color, (x, y - size//4), size//6)
        pygame.draw.circle(screen, outline_color, (x, y - size//4), size//6, 2)
        # Diagonal slash across
        pygame.draw.line(screen, outline_color, (x - size//8, y - size//3), (x + size//8, y - size//8), 3)
        # Body trapezoid
        points = [
            (x - size//6, base_y),
            (x - size//8, y - size//8),
            (x + size//8, y - size//8),
            (x + size//6, base_y)
        ]
        pygame.draw.polygon(screen, piece_color, points)
        pygame.draw.polygon(screen, outline_color, points, 2)

    elif piece_type == 'queen':
        # Crown with multiple points
        crown_points = [
            (x - size//4, base_y - size//8),
            (x - size//5, y - size//6),
            (x - size//8, y),
            (x, y - size//4),
            (x + size//8, y),
            (x + size//5, y - size//6),
            (x + size//4, base_y - size//8)
        ]
        pygame.draw.polygon(screen, piece_color, crown_points)
        pygame.draw.polygon(screen, outline_color, crown_points, 2)
        # Top circle
        pygame.draw.circle(screen, piece_color, (x, y - size//4), size//8)
        pygame.draw.circle(screen, outline_color, (x, y - size//4), size//8, 2)

    elif piece_type == 'king':
        # Cross on top
        cross_size = size // 6
        pygame.draw.line(screen, piece_color, (x, y - size//3), (x, y - size//8), 5)
        pygame.draw.line(screen, piece_color, (x - cross_size//2, y - size//4), (x + cross_size//2, y - size//4), 5)
        pygame.draw.line(screen, outline_color, (x, y - size//3), (x, y - size//8), 2)
        pygame.draw.line(screen, outline_color, (x - cross_size//2, y - size//4), (x + cross_size//2, y - size//4), 2)
        # Body
        points = [
            (x - size//4, base_y),
            (x - size//6, y),
            (x + size//6, y),
            (x + size//4, base_y)
        ]
        pygame.draw.polygon(screen, piece_color, points)
        pygame.draw.polygon(screen, outline_color, points, 2)


def get_piece_moves(piece_type, piece_color, from_row, from_col, board):
    """Get pseudo-legal moves for any piece on any board"""
    moves = []

    if piece_type == 'pawn':
        direction = -1 if piece_color == 'white' else 1
        if 0 <= from_row + direction < 8 and board[from_row + direction][from_col] is None:
            moves.append((from_row + direction, from_col))
            start_row = 6 if piece_color == 'white' else 1
            if from_row == start_row and board[from_row + 2*direction][from_col] is None:
                moves.append((from_row + 2*direction, from_col))
        for dc in [-1, 1]:
            new_row, new_col = from_row + direction, from_col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = board[new_row][new_col]
                if target and target.color != piece_color:
                    moves.append((new_row, new_col))

    elif piece_type == 'rook':
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for i in range(1, 8):
                new_row, new_col = from_row + dr*i, from_col + dc*i
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                target = board[new_row][new_col]
                if target:
                    if target.color != piece_color:
                        moves.append((new_row, new_col))
                    break
                moves.append((new_row, new_col))

    elif piece_type == 'knight':
        for dr, dc in [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]:
            new_row, new_col = from_row + dr, from_col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = board[new_row][new_col]
                if not target or target.color != piece_color:
                    moves.append((new_row, new_col))

    elif piece_type == 'bishop':
        for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            for i in range(1, 8):
                new_row, new_col = from_row + dr*i, from_col + dc*i
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                target = board[new_row][new_col]
                if target:
                    if target.color != piece_color:
                        moves.append((new_row, new_col))
                    break
                moves.append((new_row, new_col))

    elif piece_type == 'queen':
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            for i in range(1, 8):
                new_row, new_col = from_row + dr*i, from_col + dc*i
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                target = board[new_row][new_col]
                if target:
                    if target.color != piece_color:
                        moves.append((new_row, new_col))
                    break
                moves.append((new_row, new_col))

    elif piece_type == 'king':
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = from_row + dr, from_col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = board[new_row][new_col]
                    if not target or target.color != piece_color:
                        moves.append((new_row, new_col))

    return moves


class HandTracker:
    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_presence_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            sys.exit("No webcam")
        self.timestamp_ms = 0

    def get_hand_position(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb.astype(np.uint8))
        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        self.timestamp_ms += 33

        if not result.hand_landmarks:
            return None

        landmarks = result.hand_landmarks[0]
        index_x = int(landmarks[8].x * WIDTH)
        index_y = int(landmarks[8].y * HEIGHT)

        return (index_x, index_y)

    def release(self):
        self.cap.release()
        self.landmarker.close()


class ChessPiece:
    def __init__(self, piece_type, color, row, col):
        self.type = piece_type
        self.color = color
        self.row = row
        self.col = col


class ChessGame:
    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.setup_board()
        self.selected_piece = None
        self.valid_moves = []
        self.current_turn = 'white'
        self.hover_square = None
        self.hover_timer = 0
        self.captured_white = []  # Pieces captured by white (black pieces)
        self.captured_black = []  # Pieces captured by black (white pieces)
        self.in_check = False
        self.checkmate = False

    def setup_board(self):
        for col in range(8):
            self.board[1][col] = ChessPiece('pawn', 'black', 1, col)
            self.board[6][col] = ChessPiece('pawn', 'white', 6, col)

        pieces = ['rook', 'knight', 'bishop', 'queen', 'king', 'bishop', 'knight', 'rook']
        for col, piece_type in enumerate(pieces):
            self.board[0][col] = ChessPiece(piece_type, 'black', 0, col)
            self.board[7][col] = ChessPiece(piece_type, 'white', 7, col)

    def find_king(self, color, board):
        """Find the position of the king for the given color on given board"""
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece and piece.type == 'king' and piece.color == color:
                    return (row, col)
        return None

    def is_square_attacked(self, row, col, by_color, board):
        """Check if a square is attacked by any piece of the given color on given board"""
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece and piece.color == by_color:
                    moves = get_piece_moves(piece.type, piece.color, r, c, board)
                    if (row, col) in moves:
                        return True
        return False

    def is_in_check(self, color, board):
        """Check if the given color's king is in check on given board"""
        king_pos = self.find_king(color, board)
        if not king_pos:
            return False
        opponent_color = 'black' if color == 'white' else 'white'
        return self.is_square_attacked(king_pos[0], king_pos[1], opponent_color, board)

    def copy_board(self):
        """Create a deep copy of the board"""
        temp_board = [[None for _ in range(8)] for _ in range(8)]
        for r in range(8):
            for c in range(8):
                if self.board[r][c]:
                    p = self.board[r][c]
                    temp_board[r][c] = ChessPiece(p.type, p.color, r, c)
        return temp_board

    def would_be_in_check(self, piece, to_row, to_col):
        """Check if making this move would leave own king in check"""
        # Create temp board
        temp_board = self.copy_board()

        # Make the move on temp board
        from_row, from_col = piece.row, piece.col
        temp_board[from_row][from_col] = None
        temp_board[to_row][to_col] = ChessPiece(piece.type, piece.color, to_row, to_col)

        # Check if king would be in check on the temp board
        return self.is_in_check(piece.color, temp_board)

    def get_legal_moves(self, piece):
        """Get all legal moves for a piece (excluding moves that leave king in check)"""
        pseudo_moves = get_piece_moves(piece.type, piece.color, piece.row, piece.col, self.board)
        legal_moves = []

        for move in pseudo_moves:
            if not self.would_be_in_check(piece, move[0], move[1]):
                legal_moves.append(move)

        return legal_moves

    def has_legal_moves(self, color):
        """Check if the given color has any legal moves"""
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece and piece.color == color:
                    if len(self.get_legal_moves(piece)) > 0:
                        return True
        return False

    def check_game_state(self):
        """Check for check and checkmate"""
        self.in_check = self.is_in_check(self.current_turn, self.board)

        if self.in_check and not self.has_legal_moves(self.current_turn):
            self.checkmate = True
        else:
            self.checkmate = False

    def get_square_from_pos(self, x, y):
        col = (x - BOARD_OFFSET_X) // SQUARE_SIZE
        row = (y - BOARD_OFFSET_Y) // SQUARE_SIZE
        if 0 <= row < 8 and 0 <= col < 8:
            return row, col
        return None

    def update(self, pointer_pos):
        # Hand left frame - cancel selection
        if pointer_pos is None:
            self.selected_piece = None
            self.valid_moves = []
            self.hover_square = None
            self.hover_timer = 0
            return

        current_square = self.get_square_from_pos(pointer_pos[0], pointer_pos[1])

        # No valid square - reset hover
        if not current_square:
            self.hover_square = None
            self.hover_timer = 0
            return

        row, col = current_square

        # Check if this is a valid hover target
        is_valid_target = False
        if self.selected_piece:
            # If piece selected, only hover on valid moves
            is_valid_target = (row, col) in self.valid_moves
        else:
            # If no piece selected, only hover on current player's pieces
            piece = self.board[row][col]
            is_valid_target = piece and piece.color == self.current_turn

        # Not a valid target - reset hover
        if not is_valid_target:
            self.hover_square = None
            self.hover_timer = 0
            return

        # Check if square changed
        if current_square != self.hover_square:
            self.hover_square = current_square
            self.hover_timer = 0

        # Increment hover timer
        self.hover_timer += 1

        # Hover complete
        if self.hover_timer >= HOVER_TIME:
            self.hover_timer = 0

            if self.selected_piece:
                # Try to move piece
                if (row, col) in self.valid_moves:
                    # Check if capturing a piece
                    captured_piece = self.board[row][col]
                    if captured_piece:
                        if captured_piece.color == 'white':
                            self.captured_black.append(captured_piece.type)
                        else:
                            self.captured_white.append(captured_piece.type)

                    old_row, old_col = self.selected_piece.row, self.selected_piece.col
                    self.board[old_row][old_col] = None
                    self.board[row][col] = self.selected_piece
                    self.selected_piece.row = row
                    self.selected_piece.col = col
                    self.current_turn = 'black' if self.current_turn == 'white' else 'white'
                    self.selected_piece = None
                    self.valid_moves = []

                    # Check game state after move
                    self.check_game_state()
            else:
                # Try to select piece
                piece = self.board[row][col]
                if piece and piece.color == self.current_turn:
                    self.selected_piece = piece
                    self.valid_moves = self.get_legal_moves(piece)

    def draw(self, screen):
        for row in range(8):
            for col in range(8):
                x = BOARD_OFFSET_X + col * SQUARE_SIZE
                y = BOARD_OFFSET_Y + row * SQUARE_SIZE

                is_light = (row + col) % 2 == 0
                color = (240, 217, 181) if is_light else (181, 136, 99)

                if self.selected_piece and self.selected_piece.row == row and self.selected_piece.col == col:
                    color = (186, 202, 68)

                if (row, col) in self.valid_moves:
                    color = tuple(max(0, c - 30) for c in color)

                # Highlight king square if in check
                piece = self.board[row][col]
                if self.in_check and piece and piece.type == 'king' and piece.color == self.current_turn:
                    color = (220, 100, 100)

                pygame.draw.rect(screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
                pygame.draw.rect(screen, (100, 100, 100), (x, y, SQUARE_SIZE, SQUARE_SIZE), 1)

        # Draw pieces using geometric shapes
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece:
                    x = BOARD_OFFSET_X + col * SQUARE_SIZE + SQUARE_SIZE // 2
                    y = BOARD_OFFSET_Y + row * SQUARE_SIZE + SQUARE_SIZE // 2
                    draw_chess_piece(screen, piece.type, piece.color, x, y, SQUARE_SIZE - 20)

        # Draw valid move indicators
        for row, col in self.valid_moves:
            x = BOARD_OFFSET_X + col * SQUARE_SIZE + SQUARE_SIZE // 2
            y = BOARD_OFFSET_Y + row * SQUARE_SIZE + SQUARE_SIZE // 2
            pygame.draw.circle(screen, (100, 200, 100), (x, y), 10, 3)

        # Draw captured pieces
        self.draw_captured_pieces(screen)

    def draw_captured_pieces(self, screen):
        piece_size = 30
        spacing = 35

        # White's captures (black pieces) - shown on bottom (white's side)
        capture_y_white = BOARD_OFFSET_Y + BOARD_SIZE * SQUARE_SIZE + 20
        for i, piece_type in enumerate(self.captured_white):
            x = BOARD_OFFSET_X + (i % 8) * spacing + spacing // 2
            y = capture_y_white + (i // 8) * spacing
            draw_chess_piece(screen, piece_type, 'black', x, y, piece_size)

        # Black's captures (white pieces) - shown on top (black's side)
        capture_y_black = BOARD_OFFSET_Y - 40
        for i, piece_type in enumerate(self.captured_black):
            x = BOARD_OFFSET_X + (i % 8) * spacing + spacing // 2
            y = capture_y_black - (i // 8) * spacing
            draw_chess_piece(screen, piece_type, 'white', x, y, piece_size)

    def get_hover_progress(self):
        return self.hover_timer / HOVER_TIME if self.hover_timer > 0 else 0


def draw_pointer(screen, pos, hover_progress):
    if not pos:
        return

    # Base circle
    pygame.draw.circle(screen, (100, 255, 100), pos, 15, 3)
    pygame.draw.circle(screen, (100, 255, 100), pos, 3)

    # Progress arc
    if hover_progress > 0:
        radius = 20
        num_segments = int(360 * hover_progress)

        for i in range(num_segments):
            angle1 = math.radians(i - 90)
            angle2 = math.radians((i + 1) - 90)

            x1 = pos[0] + radius * math.cos(angle1)
            y1 = pos[1] + radius * math.sin(angle1)
            x2 = pos[0] + radius * math.cos(angle2)
            y2 = pos[1] + radius * math.sin(angle2)

            color_intensity = int(255 * hover_progress)
            color = (color_intensity, 255 - color_intensity // 2, 100)
            pygame.draw.line(screen, color, (x1, y1), (x2, y2), 4)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Motion Chess - Hover Controls")
    clock = pygame.time.Clock()

    game = ChessGame()
    tracker = HandTracker()

    font = pygame.font.SysFont(None, 28)
    large_font = pygame.font.SysFont(None, 40)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            running = False

        pointer_pos = tracker.get_hand_position()
        game.update(pointer_pos)

        screen.fill((50, 50, 60))
        game.draw(screen)

        draw_pointer(screen, pointer_pos, game.get_hover_progress())

        # Status text
        if game.checkmate:
            winner = 'BLACK' if game.current_turn == 'white' else 'WHITE'
            checkmate_text = f"CHECKMATE! {winner} WINS!"
            text_surface = large_font.render(checkmate_text, True, (255, 50, 50))
            text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            # Draw background box
            pygame.draw.rect(screen, (20, 20, 20), text_rect.inflate(40, 20))
            screen.blit(text_surface, text_rect)
        elif game.in_check:
            check_text = f"{game.current_turn.upper()} IS IN CHECK!"
            text_surface = large_font.render(check_text, True, (255, 100, 100))
            screen.blit(text_surface, (WIDTH // 2 - 180, HEIGHT // 2 - 200))

        turn_text = f"{game.current_turn.upper()}'s turn"
        status_text = "Hover to select/move | Leave frame to cancel"

        turn_surface = font.render(turn_text, True, (255, 255, 255))
        status_surface = font.render(status_text, True, (200, 200, 200))

        screen.blit(turn_surface, (20, 20))
        screen.blit(status_surface, (20, 50))

        if game.selected_piece:
            selected_text = f"Selected: {game.selected_piece.color} {game.selected_piece.type}"
            selected_surface = font.render(selected_text, True, (255, 255, 100))
            screen.blit(selected_surface, (20, 80))

        if pointer_pos and game.hover_timer > 0:
            progress_pct = int(game.get_hover_progress() * 100)
            progress_text = f"Hover: {progress_pct}%"
            progress_surface = font.render(progress_text, True, (100, 255, 100))
            screen.blit(progress_surface, (WIDTH - 150, 20))

        pygame.display.flip()
        clock.tick(60)

    tracker.release()
    pygame.quit()


if __name__ == "__main__":
    main()
