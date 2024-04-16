from Solver import const
import numpy as np


class SudokuBoard:
    # contructor uses string with 81 numbers[0-9] where empty field are coded as 0
    def __init__(self, board=const.TEST_BOARD, dim_x=const.BOARD_DIM_X, dim_y=const.BOARD_DIM_Y, input_type='string', cell_dim_x=const.CELL_DIM_X, cell_dim_y=const.CELL_DIM_Y):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.cell_dim_x = cell_dim_x
        self.cell_dim_y = cell_dim_y
        self.max_value = self.cell_dim_x*self.cell_dim_y
        self.number_inserted = 0
        if(input_type == 'string'):
            self.board = np.array([int(digit) for digit in board]).reshape(
                self.dim_x, self.dim_y)
        else:
            self.board = board
        self.markup = np.tile(
            np.arange(1, self.max_value+1), self.dim_x*self.dim_y).reshape(self.dim_x, self.dim_y, self.max_value)
        self.markup_depth = np.full(
            (self.dim_x, self.dim_y), self.max_value)
        self.remove_initial_values_from_markup()

    def remove_initial_values_from_markup(self):
        nonzero_indices = np.nonzero(self.board)
        for index in range(0, nonzero_indices[0].size):
            row = nonzero_indices[0][index]
            column = nonzero_indices[1][index]
            self.markup_depth[row, column] = 0
            self.markup[row, column, :] = 0
            self.number_inserted += 1
            _ = self.remove_value_from_markup(
                row, column, self.board[row, column])

    def remove_value_from_markup(self, row_index, column_index, value):
        removed_markups = []
        removed_markups += self.remove_value_from_markup_row(row_index, value)
        removed_markups += self.remove_value_from_markup_column(
            column_index, value)
        removed_markups += self.remove_value_from_markup_cell(
            row_index, column_index, value)
        return removed_markups

    def remove_value_from_markup_row(self, row_index, value):
        removed_markups = []
        for column_index in range(0, self.dim_x):
            if(self.markup[row_index, column_index, value-1] != 0):
                self.markup[row_index, column_index, value-1] = 0
                self.markup_depth[row_index, column_index] -= 1
                removed_markups += [[row_index, column_index]]
        return removed_markups

    def remove_value_from_markup_column(self, column_index, value):
        removed_markups = []
        for row_index in range(0, self.dim_y):
            if(self.markup[row_index, column_index, value-1] != 0):
                self.markup[row_index, column_index, value-1] = 0
                self.markup_depth[row_index, column_index] -= 1
                removed_markups += [[row_index, column_index]]
        return removed_markups

    def remove_value_from_markup_cell(self, row_index, column_index, value):
        removed_markups = []
        row_cell_index = (row_index//self.cell_dim_y)*self.cell_dim_y
        column_cell_index = (column_index//self.cell_dim_x)*self.cell_dim_x
        for row in range(row_cell_index, row_cell_index+self.cell_dim_y):
            for column in range(column_cell_index, column_cell_index+self.cell_dim_x):
                if(self.markup[row, column, value-1] != 0):
                    self.markup[row, column, value-1] = 0
                    self.markup_depth[row, column] -= 1
                    removed_markups += [[row, column]]
        return removed_markups

    def insert_value_into_markup(self, value, removed_markup):
        for place in removed_markup:
            self.markup[place[0], place[1], value-1] = value
            self.markup_depth[place[0], place[1]] += 1

    def solve(self):
        if(self.number_inserted == self.dim_x*self.dim_y):
            return True

        next_guess_index = np.where(self.markup_depth == np.amin(
            self.markup_depth[np.where(self.markup_depth > 0)]))
        row_index = next_guess_index[0][0]
        column_index = next_guess_index[1][0]

        values_for_check = self.markup[row_index, column_index,
                                       :][self.markup[row_index, column_index, :] > 0]

        for value in values_for_check:
            place_markups, removed_markups = self.insert_value_in_place(
                row_index, column_index, value)
            if(self.check(row_index, column_index) and self.solve()):
                return True
            else:
                self.remove_value_from_place(
                    row_index, column_index, value, removed_markups, place_markups)

        return False

    def insert_value_in_place(self, row_index, column_index, value):
        self.number_inserted += 1
        self.board[row_index, column_index] = value
        self.markup_depth[row_index, column_index] = 0
        place_markups = np.copy(self.markup[row_index, column_index, :])
        removed_markups = self.remove_value_from_markup(
            row_index, column_index, value)
        return place_markups, removed_markups

    def remove_value_from_place(self, row_index, column_index, value, removed_markups, place_markups):
        self.number_inserted -= 1
        self.board[row_index, column_index] = 0
        self.insert_value_into_markup(value, removed_markups)
        self.markup[row_index, column_index, :] = place_markups
        self.markup_depth[row_index,
                          column_index] = np.count_nonzero(place_markups)

    def check(self, row_index, column_index):
        for row in range(0, self.dim_x):
            if(self.board[row, column_index] == 0 and self.markup_depth[row, column_index] == 0):
                return False

        for column in range(0, self.dim_y):
            if(self.board[row_index, column] == 0 and self.markup_depth[row_index, column] == 0):
                return False

        row_cell_index = (row_index//self.cell_dim_x)*self.cell_dim_x
        column_cell_index = (column_index//self.cell_dim_y)*self.cell_dim_y

        for row in range(row_cell_index, row_cell_index+self.cell_dim_x):
            for column in range(column_cell_index, column_cell_index+self.cell_dim_y):
                if(self.board[row, column] == 0 and self.markup_depth[row, column] == 0):
                    return False

        return True

    def check_solution(self, str_board=const.TEST_SOLUTION):
        solution_board = np.array([int(digit) for digit in str_board]).reshape(
            self.dim_x, self.dim_y)
        if(np.array_equal(self.board, solution_board)):
            return True
        return False

    def validate_board(self):
        for index in range(0, self.dim_x):
            _, count = np.unique(self.board[index, :], return_counts=True)
            if(any(count[1:] > 1)):
                return False
            _, count = np.unique(self.board[:, index], return_counts=True)
            if(any(count[1:] > 1)):
                return False

        for column in range(0, self.dim_x, self.cell_dim_x):
            for row in range(0, self.dim_y, self.cell_dim_y):
                _, count = np.unique(self.board[row:row+self.cell_dim_y, column:column+self.cell_dim_x], return_counts=True)
                if(any(count[1:] > 1)):
                    return False

        return True