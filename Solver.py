import numpy as np
import cv2
import operator
# Class for CSP solving



class Solver:
    def __init__(self, game_info):
        self.info = game_info
        self.GRID_LEN = self.info['GRID_LEN']

        self.result = None


    def solveGame(self, game_data):
        self.data = game_data
        if self.info['game'] == 'sudoku':
            return self.solveSudoku()
        elif self.info['game'] == 'stars':
            return self.solveStars()
        elif self.info['game'] == 'cities':
            return self.solveCities()
        else:
            return None

    def solveSudoku(self):
        self.SQUARE_LEN = self.info['SQUARE_LEN']

        cells = []
        [[cells.append(str(i) + str(j)) for j in range(self.GRID_LEN)] for i in range(self.GRID_LEN)]

        domains = {}
        for var in cells:
            # var = '00'
            domains[var] = [str(k + 1) for k in range(self.GRID_LEN)]

        self.CSP = {
            "VARIABLES": cells,
            "DOMAINS": domains,
            "CONSTRAINTS": [self.alldiff_in_cols_and_rows, self.all_diff_in_areas]
        }

        initial_assignment = self.init_assignment(self.CSP)
        print('initial_assignment ', initial_assignment)
        #self.CSP['DOMAINS'] = self.update_domains(initial_assignment, domains)

        # Check initial assesment
        if self.alldiff_in_cols_and_rows(initial_assignment) or self.all_diff_in_areas(initial_assignment):
            self.print_sudoku_result(initial_assignment)
            return 'Initial assesment wrong'

        self.result = self.recursive_backtracking(initial_assignment, self.CSP)
        print(self.result)

        self.print_sudoku_result(self.result)

        return 'SOLVED'

    def solveCities(self):
        pass

    def solveStars(self):
        pass

    def is_complete(self, assignment):
        return None not in (assignment.values())

    def select_unassigned_variable(self, variables, assignment):
        nones = [var for var in variables if assignment[var] is None]
        return nones[0]

    def is_consistent(self, assignment, constraints):
        for constraint_violated in constraints:
            if constraint_violated(assignment):
                return False
        return True

    def init_assignment(self, csp):
        assignment = {}
        for var in csp["VARIABLES"]:
            assignment[var] = None

        already_found = self.data['variables_found']
        for var in already_found:
            assignment[var] = already_found[var]
        return assignment

    def recursive_backtracking(self, assignment, csp):
        if self.is_complete(assignment):
            if not self.there_are_enough_values(assignment):
                return "FAILURE"
            return assignment
        var = self.select_unassigned_variable(csp["VARIABLES"], assignment)

        domains_copy = {}
        for v in self.CSP['VARIABLES']:
            domains_copy[v] = csp["DOMAINS"][v]
        # domain = lcv_heuristic(assignment, domains_copy, var)
        domain = self.neighbors_heuristic(assignment, domains_copy, var)
        for value in domain:
            assignment[var] = value
            if self.is_consistent(assignment, csp["CONSTRAINTS"]):
                result = self.recursive_backtracking(assignment, csp)
                if result != "FAILURE":
                    return result
            assignment[var] = None
        return "FAILURE"

    def easy_inference(self, csp):
        assignment = {}

        for var in csp["VARIABLES"]:
            assignment[var] = None

        # if self.info['game'] == 'cities':
        #     asmt_matrix = np.zeros((GRID_LEN, GRID_LEN))
        #     for i in range(GRID_LEN):
        #         if observers['left'][i] == GRID_LEN:
        #             asmt_matrix[i, :] = list(range(1, GRID_LEN))
        #         if observers['left'][i] == 1:
        #             asmt_matrix[i, 0] = GRID_LEN
        #
        #         if observers['right'][i] == GRID_LEN:
        #             for k in range(1, GRID_LEN):
        #                 asmt_matrix[i, GRID_LEN - k] = k
        #         if observers['right'][i] == 1:
        #             asmt_matrix[i, GRID_LEN - 1] = GRID_LEN
        #
        #         if observers['top'][i] == GRID_LEN:
        #             asmt_matrix[:, i] = list(range(1, GRID_LEN))
        #         if observers['top'][i] == 1:
        #             asmt_matrix[0, i] = GRID_LEN
        #
        #         if observers['bottom'][i] == GRID_LEN:
        #             for k in range(1, GRID_LEN):
        #                 asmt_matrix[GRID_LEN - k, i] = k
        #         if observers['bottom'][i] == 1:
        #             asmt_matrix[GRID_LEN - 1, i] = GRID_LEN
        #
        #     asmt_flatten = asmt_matrix.flatten()
        #     for idx, var in enumerate(csp["VARIABLES"]):
        #         assignment[var] = str(int(asmt_flatten[idx])) if asmt_flatten[idx] != 0 else None

        return assignment

    def update_domains(self, definitive_asmt, domains_starting):
        domains_copy = {}
        for v in self.CSP['VARIABLES']:
            domains_copy[v] = domains_starting[v]
        for i in range(self.GRID_LEN):
            for j in range(self.GRID_LEN):
                val = definitive_asmt[str(i) + str(j)]
                if val != None:
                    for k in range(self.GRID_LEN):
                        if val in domains_copy[str(i) + str(k)]:
                            domains_copy[str(i) + str(k)].remove(val)
                        if val in domains_copy[str(k) + str(j)]:
                            domains_copy[str(k) + str(j)].remove(val)
        return domains_copy

    # Sudoku Heuristics

    def there_are_enough_values(self, assignment):
        asmt_matrix = np.array(list(assignment.values())).reshape(self.GRID_LEN, self.GRID_LEN)
        for i in range(self.GRID_LEN):
            col_values = asmt_matrix[:, i]
            row_values = asmt_matrix[i, :]
            col_ok = True
            row_ok = True
            for k in range(self.GRID_LEN):
                col_ok = col_ok and str(k + 1) in col_values
                row_ok = row_ok and str(k + 1) in row_values

            if not col_ok or not row_ok:
                return False
        return True

    def neighbors_heuristic(self, assignment, domains, var):
        domain = [d for d in domains[var]]

        asmt_matrix = np.array(list(assignment.values())).reshape(self.GRID_LEN, self.GRID_LEN)
        row = [v for v in asmt_matrix[int(var[0]), :] if v is not None]
        col = [v for v in asmt_matrix[:, int(var[1])] if v is not None]

        [domain.remove(r) for r in row if r in domain]
        [domain.remove(c) for c in col if c in domain]

        squares = []
        for i in range(self.SQUARE_LEN):
            for j in range(self.SQUARE_LEN):
                square_tmp = []
                for k in range(self.SQUARE_LEN):
                    [square_tmp.append(str(i * self.SQUARE_LEN + k) + str(s + self.SQUARE_LEN * j)) for s in range(self.SQUARE_LEN)]
                squares.append(square_tmp)

        for square in squares:
            if var in square:
                [domain.remove(v) for v in [assignment[vv] for vv in square] if v in domain]

        return domain

    def all_diff_in_areas(self, asmt):
        for i in range(self.SQUARE_LEN):
            for j in range(self.SQUARE_LEN):
                square_tmp = []
                for k in range(self.SQUARE_LEN):
                    [square_tmp.append(asmt[str(i * self.SQUARE_LEN + k) + str(s + self.SQUARE_LEN * j)]) for s in
                     range(self.SQUARE_LEN)]
                if (not None in square_tmp and not len(square_tmp) == len(set(square_tmp))):
                    return True
        return False

    def alldiff_in_cols_and_rows(self, asmt):
        asmt_matrix = np.array(list(asmt.values())).reshape(self.GRID_LEN, self.GRID_LEN)
        for i in range(self.GRID_LEN):
            row = asmt_matrix[:, i]
            col = asmt_matrix[i, :]
            if (not None in col and not len(col) == len(set(col))) or (
                    not None in row and not len(row) == len(set(row))):
                return True
        return False

    def print_sudoku_result(self, result):
        result_values = list(result.values())
        sudoku = ''
        for k in range(self.GRID_LEN * self.GRID_LEN):
            sudoku += result_values[k] + ' '
            if (k + 1) % self.SQUARE_LEN == 0:
                sudoku += '| '
            if (k + 1) % (self.SQUARE_LEN * self.SQUARE_LEN) == 0:
                sudoku += '\n'
            if (k + 1) % (self.GRID_LEN * self.SQUARE_LEN) == 0:
                sudoku += '\n'

        print(sudoku)

    def drawResult(self, grid_image):
        if self.result is not None:
            if self.info['game'] == 'sudoku':
                return self.drawSudokuResult(grid_image)

    def drawSudokuResult(self, grid_image):
        squares = []
        grid_len = self.GRID_LEN  # Ex. 9
        side = grid_image.shape[:1]
        side = side[0] / grid_len
        for j in range(grid_len):
            for i in range(grid_len):
                p1 = (int(i * side), int(j * side))  # Top left corner of a box
                p2 = (int((i + 1) * side), int((j + 1) * side))  # Bottom right corner
                squares.append((p1, p2))

        for idx, square in enumerate(squares):
            val = self.result[self.CSP['VARIABLES'][idx]]

            grid_image = cv2.putText(grid_image, str(val), (int(square[1][0]-15), int(square[1][1])-10),
                                 cv2.FONT_HERSHEY_DUPLEX, 0.8, color=(0, 255, 0))
        return grid_image
