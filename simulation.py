#simulation.py

""" Simulate Real Game Plays to use them as BaseLines"""
import numpy as np
import random
import copy
import heapq

class FruitBoxSimulator:
    def __init__(self, board_array):
        self.board = np.array(board_array, dtype=np.uint8)
        self.rows, self.cols = self.board.shape
        self.score = 0
        self.success_moves = []
        self.failed_moves = []

    def apply_move(self, move):
        try:
            (r_start, c_start), (r_end, c_end) = move
            r1, r2 = min(r_start, r_end), max(r_start, r_end)
            c1, c2 = min(c_start, c_end), max(c_start, c_end)
            section = self.board[r1:r2+1, c1:c2+1]

            if np.sum(section) != 10:
                print(f"❌ Failed (sum is not 10): move={move}, sum={np.sum(section)}")
                self.failed_moves.append(move)
                return False

            self.score += np.count_nonzero(section)
            self.board[r1:r2+1, c1:c2+1] = 0
            self.success_moves.append(move)
            return True

        except Exception:
            self.failed_moves.append(move)
            return False

    def apply_moves(self, moves):
        for move in moves:
            self.apply_move(move)

    def get_result(self):
        return {
            "score": self.score,
            "successful_moves": self.success_moves,
            "successful_moves_cnt": len(self.success_moves),
            "failed_moves": self.failed_moves,
            "failed_moves_cnt": len(self.failed_moves)
        }

    @staticmethod
    def greedy_simulation(board_array):
        """
        Core logic of the Greedy algorithm (Single Source of Truth).
        Takes a board state and returns the calculated optimal moves and score.
        """
        board = np.array(board_array, dtype=np.uint8).copy()
        rows, cols = board.shape
        
        score = 0
        moves = []
        target_number_apples = 2

        while target_number_apples <= 10:
            found_selection_in_pass = False
            break_all_loops = False
            for r in range(rows):
                for c in range(cols):
                    for r_size in range(1, rows - r + 1):
                        for c_size in range(1, cols - c + 1):
                            section = board[r:r + r_size, c:c + c_size]
                            if np.count_nonzero(section) > target_number_apples:
                                break
                            if (np.count_nonzero(section) == target_number_apples
                                and np.sum(section) == 10
                                and np.count_nonzero(section[0, :]) != 0
                                and np.count_nonzero(section[:, 0]) != 0):
                                move = ((r, c), (r + r_size - 1, c + c_size - 1))
                                moves.append(move)
                                score += target_number_apples
                                board[r:r + r_size, c:c + c_size] = 0
                                found_selection_in_pass = True
                                break_all_loops = True
                                break
                        if break_all_loops: break
                    if break_all_loops: break
                if break_all_loops: break

            if found_selection_in_pass:
                target_number_apples = 2
            else:
                target_number_apples += 1
                
        return score, moves

    def random_play(self, max_attempts=500):
        temp_board = copy.deepcopy(self.board)
        moves = []
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            # Random rectangle coordinates
            r_start = random.randint(0, self.rows - 1)
            c_start = random.randint(0, self.cols - 1)
            r_end = random.randint(r_start, min(r_start + 3, self.rows - 1))
            c_end = random.randint(c_start, min(c_start + 3, self.cols - 1))

            section = temp_board[r_start:r_end+1, c_start:c_end+1]

            # Check if valid move
            if np.all(section != 0) and np.sum(section) == 10:
                moves.append(((r_start, c_start), (r_end, c_end)))
                temp_board[r_start:r_end+1, c_start:c_end+1] = 0
                self.score += np.count_nonzero(section)
                attempts = 0  # reset attempts if a move was found

        self.success_moves = moves
        return moves
    
    # # ==================== [수정] beam_search 오류 해결 ====================
    # def beam_search(self, beam_width=5, max_depth=5):
    #     """
    #     Beam Search 알고리즘을 사용하여 최적의 이동 경로를 탐색합니다.

    #     Args:
    #         beam_width (int): 각 탐색 단계에서 유지할 후보의 수.
    #         max_depth (int): 최대 탐색 깊이.

    #     Returns:
    #         tuple: (최종 점수, 이동 경로 리스트)
    #     """
    #     initial_board = copy.deepcopy(self.board)
    #     unique_id_counter = 0

    #     # 상태(state) 자료구조: (우선순위, 누적 점수, 고유 ID, 보드 상태, 이동 경로)
    #     # 우선순위는 min-heap을 위해 음수 점수를 사용합니다.
    #     # 누적 점수는 실제 점수를 저장합니다.
    #     initial_state = (0, 0, unique_id_counter, initial_board, [])
        
    #     beam = [initial_state]
        
    #     # 탐색 전체에서 가장 점수가 높았던 상태를 저장
    #     # 초기값은 게임 시작 상태
    #     best_state_overall = initial_state

    #     for depth in range(max_depth):
    #         candidates = []

    #         for priority, accumulated_score, _, board_state, move_sequence in beam:
                
    #             # 현재 상태에서 가능한 모든 유효한 이동을 찾음
    #             found_moves = self.get_all_valid_moves(board_state)

    #             # 더 이상 이동할 수 없는 상태(경로의 끝)에 도달한 경우
    #             if not found_moves:
    #                 # 현재까지의 최고 점수 상태와 비교하여 갱신
    #                 if accumulated_score > best_state_overall[1]:
    #                     best_state_overall = (priority, accumulated_score, _, board_state, move_sequence)
    #                 continue

    #             for move in found_moves:
    #                 new_board = copy.deepcopy(board_state)
    #                 (r_start, c_start), (r_end, c_end) = move

    #                 # get_all_valid_moves에서 이미 검증되었으므로 추가 검증은 불필요
    #                 section = new_board[r_start:r_end+1, c_start:c_end+1]
                    
    #                 # 1. 점수 누적 로직 수정
    #                 move_score = np.count_nonzero(section)
    #                 new_accumulated_score = accumulated_score + move_score # 이전 점수 + 현재 점수

    #                 # 보드 상태 업데이트
    #                 new_board[r_start:r_end+1, c_start:c_end+1] = 0
    #                 new_move_sequence = move_sequence + [move]

    #                 # 휴리스틱: 미래에 가능한 움직임의 수로 미래 가치 추정
    #                 future_move_count = len(self.get_all_valid_moves(new_board))
    #                 future_score_estimate = future_move_count * 2.5  # 가중치 (조정 가능)
                    
    #                 # 우선순위 = 누적된 실제 점수 + 미래 가치 추정치
    #                 # heapq는 min-heap이므로, 점수가 높을수록 우선순위가 높도록 음수로 변환
    #                 new_priority = -(new_accumulated_score + future_score_estimate)

    #                 unique_id_counter += 1
    #                 new_state = (new_priority, new_accumulated_score, unique_id_counter, new_board, new_move_sequence)
    #                 candidates.append(new_state)

    #         if not candidates:
    #             # 모든 빔에서 더 이상 진행할 수 없으면 탐색 종료
    #             break

    #         # 우선순위(priority)가 높은 순으로 beam_width개 만큼의 후보만 남김
    #         beam = heapq.nsmallest(beam_width, candidates)

    #         # 현재 beam에서 순수 점수가 가장 높은 상태를 best_state_overall과 비교하여 갱신
    #         current_beam_best_score = max(state[1] for state in beam)
    #         if current_beam_best_score > best_state_overall[1]:
    #             # beam 내에서 최고 점수를 가진 state를 찾아서 갱신
    #             best_in_beam = max(beam, key=lambda x: x[1])
    #             best_state_overall = best_in_beam
                
    #     # 2. 최종 결과 반환 로직 수정
    #     # 탐색 과정 전체에서 가장 점수가 높았던 상태의 실제 점수와 이동 경로를 반환
    #     final_score = best_state_overall[1]
    #     final_moves = best_state_overall[4]

    #     # 시뮬레이터 내부 상태 업데이트 (선택적)
    #     self.score = final_score
    #     self.success_moves = final_moves

    #     return final_score, final_moves
    
    def get_all_valid_moves(self, board_state):
        rows, cols = board_state.shape
        valid_moves = []

        for r_size in reversed(range(1, rows + 1)):
            for c_size in reversed(range(1, cols + 1)):
                for r in range(rows - r_size + 1):
                    for c in range(cols - c_size + 1):
                        section = board_state[r:r + r_size, c:c + c_size]
                        if np.all(section != 0) and np.sum(section) == 10:
                            valid_moves.append(((r, c), (r + r_size - 1, c + c_size - 1)))
        return valid_moves