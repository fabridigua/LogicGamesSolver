import numpy as np
import cv2
import operator
# Class for CSP solving



class Solver:
    def __init__(self, game_info):
        self.info = game_info
        self.GRID_LEN = self.info['GRID_LEN']
        self.solving = False

        self.result = None


    def solveGame(self, game_data):
        self.data = game_data
        self.iterations = 0
        if self.info['game'] == 'sudoku':
            return self.solveSudoku()
        elif self.info['game'] == 'stars':
            return self.solveStars()
        elif self.info['game'] == 'skyscrapers':
            return self.solveSkyscrapers()
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

        #initial_assignment = self.init_assignment(self.CSP)
        initial_assignment = self.easy_inference(self.CSP)
        print('initial_assignment ', initial_assignment)
        #self.CSP['DOMAINS'] = self.update_domains(initial_assignment, domains)

        # Check initial assesment
        if self.alldiff_in_cols_and_rows(initial_assignment) or self.all_diff_in_areas(initial_assignment):
            self.print_sudoku_result(initial_assignment)
            return 'WRONG_INITIAL_ASSIGNMENT'

        self.solving = True
        self.result = self.recursive_backtracking(initial_assignment, self.CSP)
        self.solving = False

        print('self.iterations ', self.iterations)
        print(self.result)
        if self.result != "FAILURE":
            self.print_sudoku_result(self.result)
            return 'SOLVED'

        return 'FAILURE'

    def solveSkyscrapers(self):
        observers = {}

        observers["left"] = [int(self.data['variables_found'][str(k)+'0']) for k in range(1, self.GRID_LEN+1)]
        observers["right"] = [int(self.data['variables_found'][str(k)+str(self.GRID_LEN+1)]) for k in range(1, self.GRID_LEN+1)]
        observers["top"] = [int(self.data['variables_found']['0'+str(k)]) for k in range(1, self.GRID_LEN+1)]
        observers["bottom"] = [int(self.data['variables_found'][str(self.GRID_LEN+1) + str(k)]) for k in range(1, self.GRID_LEN+1)]

        self.observers = observers

        cells = []
        [[cells.append(str(i) + str(j)) for j in range(self.GRID_LEN)] for i in range(self.GRID_LEN)]

        domains = {}
        for var in cells:
            # var = '00'
            domains[var] = [str(k + 1) for k in range(self.GRID_LEN)]

        self.CSP = {
            "VARIABLES": cells,
            "DOMAINS": domains,
            "CONSTRAINTS": [self.alldiff_in_cols_and_rows, self.values_are_ordered]
        }

        # initial_assignment = self.init_assignment(self.CSP)
        initial_assignment = self.easy_inference(self.CSP)
        print('initial_assignment ', initial_assignment)
        self.CSP['DOMAINS'] = self.update_domains(initial_assignment, domains)

        # Check initial assesment
        if self.alldiff_in_cols_and_rows(initial_assignment) or self.values_are_ordered(initial_assignment):
            return 'WRONG_INITIAL_ASSIGNMENT'

        self.solving = True
        self.result = self.recursive_backtracking(initial_assignment, self.CSP)
        self.solving = False

        print(self.result)
        print('self.iterations ', self.iterations)
        if self.result != "FAILURE":
            # self.print_stars_result(self.result)
            return 'SOLVED'

        return 'FAILURE'

    def solveStars(self):
        self.NUM_STARS = self.info['NUM_STARS']

        cells = []
        [[cells.append(str(i) + str(j)) for j in range(self.GRID_LEN)] for i in range(self.GRID_LEN)]

        domains = {}
        for var in cells:
            # var = '00'
            domains[var] = ['0', '1']

        self.CSP = {
            "VARIABLES": cells,
            "DOMAINS": domains,
            "CONSTRAINTS": [self.only_X_in_colums_and_rows, self.only_x_in_areas, self.never_adjacents, self.max_X_zeros]
        }

        # initial_assignment = self.init_assignment(self.CSP)
        initial_assignment = self.easy_inference(self.CSP)
        print('initial_assignment ', initial_assignment)
        #self.CSP['DOMAINS'] = self.update_domains(initial_assignment, domains)

        # Check initial assesment
        if self.only_X_in_colums_and_rows(initial_assignment) or self.only_x_in_areas(initial_assignment) or self.never_adjacents(initial_assignment):
            return 'WRONG_INITIAL_ASSIGNMENT'

        self.solving = True
        self.result = self.recursive_backtracking(initial_assignment, self.CSP)
        self.solving = False

        print(self.result)
        print('self.iterations ',self.iterations)
        if self.result != "FAILURE":
            # self.print_stars_result(self.result)
            return 'SOLVED'

        return 'FAILURE'

    def is_complete(self, assignment):
        return None not in (assignment.values())

    def select_unassigned_variable(self, variables, assignment):
        self.iterations += 1
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
        if self.info['game'] == 'sudoku':
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

        if self.info['game'] == 'stars':
            for area in self.data['variables_found']:
                for k in range(self.GRID_LEN):
                    if len(area) == len([el for el in area if el[0] == str(k)]):
                        # all in row
                        for r in range(self.GRID_LEN):
                            if str(k) + str(r) not in area:
                                assignment[str(k) + str(r)] = '0'
                                csp['DOMAINS'][str(k) + str(r)] = []
                    if len(area) == len([el for el in area if el[1] == str(k)]):
                        # all in col
                        for r in range(self.GRID_LEN):
                            if str(r) + str(k) not in area:
                                assignment[str(r) + str(k)] = '0'
                                csp['DOMAINS'][str(r) + str(k)] = []
                    if self.NUM_STARS == 1:
                        # if all elements in the row are inside the area, set to 0 other area elements
                        row = [str(k) + str(i) for i in range(self.GRID_LEN)]
                        if self.GRID_LEN == len([r for r in row if r in area]):
                            for el in [e for e in area if e not in row]:
                                assignment[el] = '0'
                                csp['DOMAINS'][el] = []
                        # if all elements in the col are inside the area, set to 0 other area elements
                        col = [str(i) + str(k) for i in range(self.GRID_LEN)]
                        if self.GRID_LEN == len([c for c in col if c in area]):
                            for el in [e for e in area if e not in col]:
                                assignment[el] = '0'
                                csp['DOMAINS'][el] = []

            # If an area has only one cell empty, fill with 1 and set to 0 corrisponding row/col
            if self.NUM_STARS == 1:
                for area in self.data['variables_found']:
                    area_nones = [el for el in area if assignment[el] is None]
                    if len(area_nones) == 1:
                        row = [area_nones[0][0] + str(k) for k in range(self.GRID_LEN) if
                               area_nones[0][0] + str(k) is not area_nones[0]]
                        col = [str(k) + area_nones[0][1] for k in range(self.GRID_LEN) if
                               str(k) + area_nones[0][1] is not area_nones[0]]
                        for r in row:
                            assignment[r] = '0'
                            csp['DOMAINS'][r] = []
                        for c in col:
                            assignment[c] = '0'
                            csp['DOMAINS'][c] = []
                        assignment[area_nones[0]] = '1'
                        csp['DOMAINS'][area_nones[0]] = []

            # If an row/col has only one cell empty, fill with 1 and set to 0 corrisponding row/col
            if self.NUM_STARS == 1:
                for k in range(self.GRID_LEN):
                    row_nones = [str(k) + str(r) for r in range(self.GRID_LEN) if assignment[str(k) + str(r)] is None]
                    col_nones = [str(c) + str(k) for c in range(self.GRID_LEN) if assignment[str(c) + str(k)] is None]
                    if len(row_nones) == 1:
                        assignment[row_nones[0]] = '1'
                        csp['DOMAINS'][row_nones[0]] = []
                    if len(col_nones) == 1:
                        assignment[col_nones[0]] = '1'
                        csp['DOMAINS'][col_nones[0]] = []
            # For each 1s put 0 in diagonals also
            for var in [v for v in csp['VARIABLES'] if assignment[v] == '1']:
                e_0 = int(var[0])
                e_1 = int(var[1])
                diagonals = [str(e_0 - 1) + str(e_1 - 1), str(e_0 - 1) + str(e_1 + 1), str(e_0 + 1) + str(e_1 - 1),
                             str(e_0 + 1) + str(e_1 + 1)]
                for d in diagonals:
                    if d in csp['VARIABLES']:
                        assignment[d] = '0'
                        csp['DOMAINS'][d] = []


        elif self.info['game'] == 'skyscrapers':
            asmt_matrix = np.zeros((self.GRID_LEN, self.GRID_LEN))
            for i in range(self.GRID_LEN):
                if self.observers['left'][i] == self.GRID_LEN:
                    asmt_matrix[i, :] = list(range(1, self.GRID_LEN+1))
                if self.observers['left'][i] == 1:
                    asmt_matrix[i, 0] = self.GRID_LEN

                if self.observers['right'][i] == self.GRID_LEN:
                    for k in range(1, self.GRID_LEN+1):
                        asmt_matrix[i, self.GRID_LEN - k] = k
                if self.observers['right'][i] == 1:
                    asmt_matrix[i, self.GRID_LEN - 1] = self.GRID_LEN

                if self.observers['top'][i] == self.GRID_LEN:
                    asmt_matrix[:, i] = list(range(1, self.GRID_LEN+1))
                if self.observers['top'][i] == 1:
                    asmt_matrix[0, i] = self.GRID_LEN

                if self.observers['bottom'][i] == self.GRID_LEN:
                    for k in range(1, self.GRID_LEN+1):
                        asmt_matrix[self.GRID_LEN - k, i] = k
                if self.observers['bottom'][i] == 1:
                    asmt_matrix[self.GRID_LEN - 1, i] = self.GRID_LEN

            asmt_flatten = asmt_matrix.flatten()
            for idx, var in enumerate(csp["VARIABLES"]):
                assignment[var] = str(int(asmt_flatten[idx])) if asmt_flatten[idx] != 0 else None
        elif self.info['game'] == 'sudoku':
            already_found = self.data['variables_found']
            for var in already_found:
                assignment[var] = already_found[var]
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

    # Stars
    def only_x_in_areas(self, asmt):
        for area in self.data['variables_found']:
            area_stars = [asmt[variable] for variable in area if asmt[variable] == "1"]
            if not len(area_stars) <= self.NUM_STARS:
                return True
        return False

    def never_adjacents(self, asmt):
        for i in range(self.GRID_LEN):
            for j in range(self.GRID_LEN):
                if (asmt[str(i) + str(j)] != "1"):
                    continue
                for v in range(i - 1, i + 2):
                    for u in range(j - 1, j + 2):
                        if v is i and u is j:
                            continue
                        if str(v) + str(u) in self.CSP["VARIABLES"]:
                            if asmt[str(v) + str(u)] == "1":
                                return True
        return False

    def only_X_in_colums_and_rows(self, asmt):
        for i in range(self.GRID_LEN):
            col = [str(i) + str(j) for j in range(self.GRID_LEN)]
            row = [str(j) + str(i) for j in range(self.GRID_LEN)]
            col_stars = [asmt[variable] for variable in col if asmt[variable] == "1"]
            row_stars = [asmt[variable] for variable in row if asmt[variable] == "1"]
            if not len(col_stars) <= self.NUM_STARS:
                return True
            if not len(row_stars) <= self.NUM_STARS:
                return True
        return False

    def max_X_zeros(self, asmt):
        for i in range(self.GRID_LEN):
            col = [str(i) + str(j) for j in range(self.GRID_LEN)]
            row = [str(j) + str(i) for j in range(self.GRID_LEN)]
            col_zeros = [variable for variable in col if asmt[variable] == "0"]
            row_zeros = [variable for variable in row if asmt[variable] == "0"]
            if len(col_zeros) == self.GRID_LEN:
                return True
            if len(row_zeros) == self.GRID_LEN:
                return True
        return False

    # Sudoku Heuristics

    def there_are_enough_values(self, assignment):
        if self.info['game'] == 'sudoku' or self.info['game'] == 'skyscrapers':
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
        elif self.info['game'] == 'stars':
            for i in range(self.GRID_LEN):
                col = [str(i) + str(j) for j in range(self.GRID_LEN)]
                row = [str(j) + str(i) for j in range(self.GRID_LEN)]
                col_stars = [assignment[variable] for variable in col if assignment[variable] == "1"]
                row_stars = [assignment[variable] for variable in row if assignment[variable] == "1"]
                if not len(col_stars) == self.NUM_STARS or not len(row_stars) == self.NUM_STARS:
                    return False

        return True

    def neighbors_heuristic(self, assignment, domains, var):
        domain = [d for d in domains[var]]

        if self.info['game'] == 'sudoku':
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
                        [square_tmp.append(str(i * self.SQUARE_LEN + k) + str(s + self.SQUARE_LEN * j)) for s in
                         range(self.SQUARE_LEN)]
                    squares.append(square_tmp)

            for square in squares:
                if var in square:
                    [domain.remove(v) for v in [assignment[vv] for vv in square] if v in domain]
        elif self.info['game'] == 'stars':
            asmt_matrix = np.array(list(assignment.values())).reshape(self.GRID_LEN, self.GRID_LEN)
            row = [v for v in asmt_matrix[int(var[0]), :] if v is not None]
            col = [v for v in asmt_matrix[:, int(var[1])] if v is not None]

            if '1' in domain:
                if '1' in row:
                    domain.remove('1')
                elif '1' in col:
                    domain.remove('1')
                else:
                    for square in self.data['variables_found']:
                        if var in square and '1' in [assignment[vv] for vv in square]:
                            domain.remove('1')

        elif self.info['game'] == 'skyscrapers':
            domain = [d for d in domains[var]]

            asmt_matrix = np.array(list(assignment.values())).reshape(self.GRID_LEN, self.GRID_LEN)
            row = [v for v in asmt_matrix[int(var[0]), :] if v is not None]
            col = [v for v in asmt_matrix[:, int(var[1])] if v is not None]

            [domain.remove(r) for r in row if r in domain]
            [domain.remove(c) for c in col if c in domain]


        return domain

    def all_diff_in_areas(self, asmt):
        for i in range(self.SQUARE_LEN):
            for j in range(self.SQUARE_LEN):
                square_tmp = []
                for k in range(self.SQUARE_LEN):
                    [square_tmp.append(asmt[str(i * self.SQUARE_LEN + k) + str(s + self.SQUARE_LEN * j)]) for s in
                     range(self.SQUARE_LEN)]
                if (not None in square_tmp and not len(square_tmp) == len(set(square_tmp))):# or (self.solving and not len(square_tmp) == len(set(square_tmp))):
                    return True
        return False

    def alldiff_in_cols_and_rows(self, asmt):
        asmt_matrix = np.array(list(asmt.values())).reshape(self.GRID_LEN, self.GRID_LEN)
        for i in range(self.GRID_LEN):
            row = asmt_matrix[:, i]
            col = asmt_matrix[i, :]
            if (not None in col and not len(col) == len(set(col))) or (
                    not None in row and not len(row) == len(set(row))):# or (self.solving and (not len(col) == len(set(col)) or not len(row) == len(set(row)))):
                return True
        return False

    def print_sudoku_result(self, result):
        result_values = list(result.values())
        sudoku = ''
        for k in range(self.GRID_LEN * self.GRID_LEN):
            if result_values[k] is not None:
                sudoku += result_values[k] + ' '
                if (k + 1) % self.SQUARE_LEN == 0:
                    sudoku += '| '
                if (k + 1) % (self.SQUARE_LEN * self.SQUARE_LEN) == 0:
                    sudoku += '\n'
                if (k + 1) % (self.GRID_LEN * self.SQUARE_LEN) == 0:
                    sudoku += '\n'

        print(sudoku)

    def drawResult(self, grid_image, data=None):
        if self.result is not None:
            if self.info['game'] == 'sudoku':
                return self.drawSudokuResult(grid_image, data)
            if self.info['game'] == 'stars':
                return self.drawStarsResult(grid_image, data)
            if self.info['game'] == 'skyscrapers':
                return self.drawSkyscrapersResult(grid_image, data)

    def drawSudokuResult(self, grid_image, sudoku_values):
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
            if self.CSP['VARIABLES'][idx] in sudoku_values:
                val = sudoku_values[self.CSP['VARIABLES'][idx]]

                grid_image = cv2.putText(grid_image, str(val), (int(square[1][0]-30), int(square[1][1])-10),
                                     cv2.FONT_HERSHEY_DUPLEX, 0.8, color=(0, 255, 0))
        return grid_image

    def drawStarsResult(self, grid_image, data):
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
            if self.CSP['VARIABLES'][idx] in data:
                val = data[self.CSP['VARIABLES'][idx]]

                if val == '0':
                    grid_image = cv2.putText(grid_image, str(val), (int(square[1][0]-15), int(square[1][1])-10),
                                         cv2.FONT_HERSHEY_DUPLEX, 0.4, color=(0, 0, 255))
                else:
                    grid_image = cv2.putText(grid_image, str(val), (int(square[1][0]-22), int(square[1][1])-7),
                                         cv2.FONT_HERSHEY_DUPLEX, 0.7, color=(0, 255, 0))
        return grid_image

    def values_are_ordered(self, asmt):

        LEFT = self.observers["left"]
        RIGHT = self.observers["right"]
        TOP = self.observers["top"]
        BOTTOM = self.observers["bottom"]

        # asmt_values = list(asmt.values())
        asmt_matrix = np.array(list(asmt.values())).reshape(self.GRID_LEN, self.GRID_LEN)
        asmt_matrix = np.where(asmt_matrix == None, 0, asmt_matrix).astype(int)

        for i in range(self.GRID_LEN):
            # i = 0
            # LEFT => 00 01 02 03
            l_row = asmt_matrix[i, :]
            # RIGHT => 03 02 01 00
            r_row = l_row[::-1]
            # TOP => 00 10 20 30
            t_row = asmt_matrix[:, i]
            # BOTTOM => 30 20 10 00
            b_row = t_row[::-1]

            if 0 in l_row or 0 in t_row:
                continue

            left_visibles = 1
            right_visibles = 1
            top_visibles = 1
            bottom_visibles = 1
            for k in range(1, self.GRID_LEN):
                left_visibles = left_visibles + 1 if l_row[k] > 0 and l_row[k] > max(l_row[:k]) else left_visibles
                right_visibles = right_visibles + 1 if r_row[k] > 0 and r_row[k] > max(r_row[:k]) else right_visibles
                top_visibles = top_visibles + 1 if t_row[k] > 0 and t_row[k] > max(t_row[:k]) else top_visibles
                bottom_visibles = bottom_visibles + 1 if b_row[k] > 0 and b_row[k] > max(b_row[:k]) else bottom_visibles

            if not (LEFT[i] > 0 and LEFT[i] == left_visibles) or not (RIGHT[i] > 0 and RIGHT[i] == right_visibles) or not (
                    TOP[i] > 0 and TOP[i] == top_visibles) or not (BOTTOM[i] > 0 and BOTTOM[i] == bottom_visibles):
                return True
        return False

    def drawSkyscrapersResult(self, grid_image, data):
        squares = []
        grid_len = self.GRID_LEN  # Ex. 9
        side = grid_image.shape[:1]
        side = side[0] / (grid_len + 2)
        for j in range(grid_len):
            for i in range(grid_len):
                p1 = (int(i * side), int(j * side))  # Top left corner of a box
                p2 = (int((i + 1) * side), int((j + 1) * side))  # Bottom right corner
                squares.append((p1, p2))

        for idx, square in enumerate(squares):
            if self.CSP['VARIABLES'][idx] in data:
                val = data[self.CSP['VARIABLES'][idx]]

                grid_image = cv2.putText(grid_image, str(val), (int(square[1][0])+10, int(square[1][1])+30),
                                     cv2.FONT_HERSHEY_DUPLEX, 0.8, color=(0, 255, 0))
        return grid_image