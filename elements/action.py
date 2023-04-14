"""Contains a class which represents a certain action"""


class Action:
    def __init__(self, no_of_stones: int, pile: int, q_value: int = None) -> None:
        if no_of_stones < 0:
            raise ValueError("Number of stones should be greater-equal 1.")
        self.pile = pile
        self.no_of_stones = no_of_stones
        self.q_value = q_value

    def __str__(self) -> str:
        return f"({self.no_of_stones}, {self.pile}): {self.q_value:.2f}"

    def __repr__(self) -> str:
        return f"({self.no_of_stones}, {self.pile}): {self.q_value:.2f}"

    def __eq__(self, __o: object) -> bool:
        return self.pile == __o.pile and self.no_of_stones == __o.no_of_stones
