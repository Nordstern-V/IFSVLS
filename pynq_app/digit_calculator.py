#!/usr/bin/env python3
"""Small, hardware-independent calculator for confirmed D-mode tokens.

Each operand is one digit (0..9).  Operator labels match the recognition
cache: plus, sub, mul and div.  Intermediate values use exact fractions;
only the LCD result text may be rounded, explicitly marked with ``~``.
"""

from fractions import Fraction
from numbers import Integral


OPERATOR_SYMBOLS = {"plus": "+", "sub": "-", "mul": "*", "div": "/"}
MAX_NUMBERS = 10
MAX_OPERATORS = 9


def _result_text(value):
    """Format a Fraction using at most six decimal places for the LCD."""
    if value.denominator == 1:
        return str(value.numerator)
    # Integer arithmetic makes display rounding independent of float error.
    scale = 10 ** 6
    scaled = abs(value.numerator) * scale
    rounded, remainder = divmod(scaled, value.denominator)
    if 2 * remainder >= value.denominator:
        rounded += 1
    whole, fractional = divmod(rounded, scale)
    text = "{}.{:06d}".format(whole, fractional).rstrip("0").rstrip(".")
    if value.numerator < 0 and rounded:
        text = "-" + text
    if rounded * value.denominator != scaled:
        text = "~" + text
    return text


class DigitCalculator:
    """Store up to ten digits/nine operators and evaluate with precedence.

    Mutation methods validate before changing any state.  Invalid additions
    raise ValueError and preserve the existing expression and result.  A
    failed evaluation records an LCD error without discarding the tokens.
    """

    def __init__(self):
        self._tokens = []
        self.result = None
        self.last_error = None

    @property
    def tokens(self):
        """An immutable snapshot, alternating integer digits/operator labels."""
        return tuple(self._tokens)

    @property
    def number_count(self):
        return (len(self._tokens) + 1) // 2

    @property
    def operator_count(self):
        return len(self._tokens) // 2

    @property
    def expecting(self):
        if self.number_count == MAX_NUMBERS:
            return None
        return "digits" if len(self._tokens) % 2 == 0 else "operators"

    @property
    def can_evaluate(self):
        return bool(self._tokens) and len(self._tokens) % 2 == 1

    @property
    def result_text(self):
        if self.last_error is not None:
            return self.last_error
        return "" if self.result is None else _result_text(self.result)

    def _invalidate_result(self):
        self.result = None
        self.last_error = None

    def append_number(self, value):
        # Recognition supplies a string; tests and other callers may use int.
        if isinstance(value, str):
            if len(value) != 1 or value not in "0123456789":
                raise ValueError("number must be a single digit 0..9")
            number = int(value)
        elif isinstance(value, Integral) and not isinstance(value, bool):
            number = int(value)
            if not 0 <= number <= 9:
                raise ValueError("number must be a single digit 0..9")
        else:
            raise ValueError("number must be a single digit 0..9")
        if self.number_count >= MAX_NUMBERS:
            raise ValueError("maximum {} numbers".format(MAX_NUMBERS))
        if self.expecting != "digits":
            raise ValueError("operator required before next number")
        self._tokens.append(number)
        self._invalidate_result()

    def append_operator(self, label):
        if not isinstance(label, str) or label not in OPERATOR_SYMBOLS:
            raise ValueError("operator must be plus, sub, mul or div")
        if self.operator_count >= MAX_OPERATORS:
            raise ValueError("maximum {} operators".format(MAX_OPERATORS))
        if self.expecting != "operators":
            raise ValueError("number required before operator")
        self._tokens.append(label)
        self._invalidate_result()

    def undo(self):
        """Remove and return the final token, or None when already empty."""
        if not self._tokens:
            return None
        removed = self._tokens.pop()
        self._invalidate_result()
        return removed

    def reset(self):
        self._tokens.clear()
        self._invalidate_result()

    def display_expression(self):
        return " ".join(
            OPERATOR_SYMBOLS[token] if isinstance(token, str) else str(token)
            for token in self._tokens
        )

    def _evaluation_error(self, message):
        self.result = None
        self.last_error = message
        raise ValueError(message)

    def evaluate(self):
        """Evaluate the confirmed tokens, preserving the expression itself."""
        if not self._tokens:
            self._evaluation_error("EMPTY")
        if not self.can_evaluate:
            self._evaluation_error("INCOMPLETE")

        total = Fraction(0)
        term = Fraction(self._tokens[0])
        sign = 1
        for index in range(1, len(self._tokens), 2):
            label = self._tokens[index]
            right = Fraction(self._tokens[index + 1])
            if label == "mul":
                term *= right
            elif label == "div":
                if right == 0:
                    self._evaluation_error("DIV/0")
                term /= right
            else:
                total += sign * term
                sign = 1 if label == "plus" else -1
                term = right
        self.result = total + sign * term
        self.last_error = None
        return self.result
