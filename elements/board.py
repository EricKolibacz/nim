"""Defining board positions and related information"""
import torch


class Board:
    def __init__(self, position: list) -> None:
        self.position = position
        self.no_of_piles = len(position)
        self.binary_reprensation = self.transform_board_to_binary(position)

    def transform_board_to_binary(self, board_position: list) -> int:
        """Transforms a board position into a unique category.

        Args:
            board (list): board position

        Returns:
            int: category
        """
        bin_seq = []
        for i in range(self.no_of_piles):
            # convert the number of sticks in each heap in a 3-bit binary representation
            binary_string = format(board_position[i], "b").zfill(3)
            for i in binary_string:
                bin_seq.append(int(i))
        return bin_seq

    def __hash__(self) -> int:
        return int(str.encode("".join(map(str, self.binary_reprensation))), base=2)

    def __eq__(self, __o: object) -> bool:
        return self.__hash__() == __o.__hash__()

    def __repr__(self) -> str:
        return f"{self.position}"
