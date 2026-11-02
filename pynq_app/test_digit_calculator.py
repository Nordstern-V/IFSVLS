#!/usr/bin/env python3
"""Standard-library tests for confirmed single-character expressions."""

from fractions import Fraction
import unittest

from digit_calculator import DigitCalculator


class DigitCalculatorTests(unittest.TestCase):
    @staticmethod
    def expression(*tokens):
        calculator = DigitCalculator()
        for index, token in enumerate(tokens):
            if index % 2 == 0:
                calculator.append_number(token)
            else:
                calculator.append_operator(token)
        return calculator

    def test_arithmetic_precedence_and_left_associativity(self):
        cases = (
            ((2, "plus", 3), 5),
            ((2, "plus", 3, "mul", 4), 14),
            ((8, "div", 2, "mul", 3), 12),
            ((8, "div", 4, "mul", 2), 4),
            ((9, "sub", 3, "sub", 2), 4),
            ((8, "div", 2, "div", 2), 2),
            ((2, "sub", 5), -3),
            ((9, "sub", 8, "div", 2, "plus", 3, "mul", 4), 17),
            ((1, "plus", 2, "mul", 3, "sub", 4, "div", 2), 5),
            ((0, "mul", 9, "plus", 1), 1),
        )
        for tokens, expected in cases:
            with self.subTest(tokens=tokens):
                calculator = self.expression(*tokens)
                before = calculator.tokens
                self.assertEqual(calculator.evaluate(), Fraction(expected))
                self.assertIsInstance(calculator.result, Fraction)
                self.assertEqual(calculator.tokens, before)
                self.assertEqual(calculator.evaluate(), Fraction(expected))

    def test_single_digit_and_zero_are_complete_expressions(self):
        calculator = DigitCalculator()
        self.assertFalse(calculator.can_evaluate)
        self.assertEqual(calculator.expecting, "digits")
        calculator.append_number("0")
        self.assertEqual(calculator.tokens, (0,))
        self.assertTrue(calculator.can_evaluate)
        self.assertEqual(calculator.evaluate(), Fraction(0))
        self.assertEqual(calculator.result_text, "0")
        self.assertEqual(calculator.expecting, "operators")

    def test_exact_fraction_and_display_rounding(self):
        cases = (
            ((7, "div", 2), Fraction(7, 2), "3.5"),
            ((1, "div", 3), Fraction(1, 3), "~0.333333"),
            ((2, "div", 3), Fraction(2, 3), "~0.666667"),
            ((0, "sub", 2, "div", 3), Fraction(-2, 3), "~-0.666667"),
            ((1, "div", 8), Fraction(1, 8), "0.125"),
            ((1, "div", 8, "div", 8, "div", 2), Fraction(1, 128), "~0.007813"),
            ((9, "mul", 9, "mul", 9, "mul", 9, "mul", 9), Fraction(59049), "59049"),
        )
        for tokens, expected, text in cases:
            with self.subTest(tokens=tokens):
                calculator = self.expression(*tokens)
                self.assertEqual(calculator.evaluate(), expected)
                self.assertEqual(calculator.result_text, text)

    def test_invalid_input_is_atomic_and_preserves_result(self):
        calculator = self.expression(2, "plus", 3)
        calculator.evaluate()
        before = (calculator.tokens, calculator.result, calculator.result_text,
                  calculator.last_error)
        for invalid in (-1, 10, 2.0, True, None, "", "12", " 2", "２", [], Fraction(2)):
            with self.subTest(number=invalid):
                with self.assertRaises(ValueError):
                    calculator.append_number(invalid)
                self.assertEqual((calculator.tokens, calculator.result,
                                  calculator.result_text, calculator.last_error), before)
        for invalid in ("+", "pow", "__import__", 1, None, []):
            with self.subTest(operator=invalid):
                with self.assertRaises(ValueError):
                    calculator.append_operator(invalid)
                self.assertEqual((calculator.tokens, calculator.result,
                                  calculator.result_text, calculator.last_error), before)

    def test_alternating_order_rejects_duplicate_types(self):
        calculator = DigitCalculator()
        with self.assertRaises(ValueError):
            calculator.append_operator("plus")
        self.assertEqual(calculator.tokens, ())
        calculator.append_number(2)
        calculator.evaluate()
        with self.assertRaises(ValueError):
            calculator.append_number(3)
        self.assertEqual(calculator.tokens, (2,))
        self.assertEqual(calculator.result, Fraction(2))
        calculator.append_operator("plus")
        with self.assertRaises(ValueError):
            calculator.append_operator("sub")
        self.assertEqual(calculator.tokens, (2, "plus"))

    def test_five_numbers_can_be_extended(self):
        calculator = self.expression(1, "plus", 2, "plus", 3, "plus", 4, "plus", 5)
        self.assertEqual(calculator.number_count, 5)
        self.assertEqual(calculator.operator_count, 4)
        self.assertEqual(calculator.expecting, "operators")
        self.assertEqual(calculator.evaluate(), Fraction(15))
        calculator.append_operator("plus")
        calculator.append_number(6)
        self.assertEqual(calculator.number_count, 6)
        self.assertEqual(calculator.evaluate(), Fraction(21))

    def test_ten_number_nine_operator_limit(self):
        calculator = self.expression(
            1, "plus", 2, "plus", 3, "plus", 4, "plus", 5,
            "plus", 6, "plus", 7, "plus", 8, "plus", 9, "plus", 0,
        )
        self.assertEqual(calculator.number_count, 10)
        self.assertEqual(calculator.operator_count, 9)
        self.assertIsNone(calculator.expecting)
        self.assertTrue(calculator.can_evaluate)
        self.assertEqual(calculator.evaluate(), Fraction(45))
        before = (calculator.tokens, calculator.result, calculator.result_text)
        with self.assertRaisesRegex(ValueError, "maximum 9 operators"):
            calculator.append_operator("plus")
        with self.assertRaisesRegex(ValueError, "maximum 10 numbers"):
            calculator.append_number(6)
        self.assertEqual((calculator.tokens, calculator.result, calculator.result_text), before)
        self.assertEqual(calculator.undo(), 0)
        self.assertEqual(calculator.expecting, "digits")
        with self.assertRaisesRegex(ValueError, "maximum 9 operators"):
            calculator.append_operator("plus")
        calculator.append_number(9)
        self.assertIsNone(calculator.expecting)
        self.assertEqual(calculator.evaluate(), Fraction(54))

    def test_ten_number_multiplication_keeps_large_integer_exact(self):
        calculator = self.expression(*([9] + ["mul", 9] * 9))
        self.assertEqual(calculator.number_count, 10)
        self.assertEqual(calculator.operator_count, 9)
        self.assertEqual(calculator.evaluate(), Fraction(3486784401))
        self.assertEqual(calculator.result_text, "3486784401")

    def test_mutation_clears_stale_result_and_undo_tracks_slots(self):
        calculator = self.expression(2, "plus", 3)
        self.assertEqual(calculator.display_expression(), "2 + 3")
        snapshot = calculator.tokens
        self.assertIsInstance(snapshot, tuple)
        calculator.evaluate()
        calculator.append_operator("mul")
        self.assertIsNone(calculator.result)
        self.assertEqual(calculator.result_text, "")
        self.assertIsNone(calculator.last_error)
        self.assertEqual(calculator.expecting, "digits")
        self.assertFalse(calculator.can_evaluate)
        self.assertEqual(calculator.undo(), "mul")
        self.assertEqual(calculator.tokens, snapshot)
        self.assertEqual(calculator.expecting, "operators")
        self.assertEqual(calculator.evaluate(), Fraction(5))
        self.assertEqual(calculator.undo(), 3)
        self.assertIsNone(calculator.result)
        self.assertEqual(calculator.tokens, (2, "plus"))
        self.assertEqual((calculator.number_count, calculator.operator_count), (1, 1))
        calculator.reset()
        self.assertEqual(calculator.tokens, ())
        self.assertEqual(calculator.display_expression(), "")
        self.assertEqual(calculator.result_text, "")
        self.assertEqual((calculator.number_count, calculator.operator_count), (0, 0))
        self.assertIsNone(calculator.undo())
        self.assertEqual(snapshot, (2, "plus", 3))

    def test_division_by_zero_retains_tokens_and_can_be_corrected(self):
        for tokens in ((8, "div", 0), (0, "mul", 8, "div", 0)):
            with self.subTest(tokens=tokens):
                calculator = self.expression(*tokens)
                with self.assertRaisesRegex(ValueError, "^DIV/0$"):
                    calculator.evaluate()
                self.assertEqual(calculator.tokens, tokens)
                self.assertIsNone(calculator.result)
                self.assertEqual(calculator.last_error, "DIV/0")
                self.assertEqual(calculator.result_text, "DIV/0")
                with self.assertRaises(ValueError):
                    calculator.append_number(2)
                self.assertEqual(calculator.result_text, "DIV/0")
                self.assertEqual(calculator.undo(), 0)
                self.assertEqual(calculator.result_text, "")
                self.assertIsNone(calculator.last_error)
                calculator.append_number(2)
                self.assertEqual(calculator.evaluate(), Fraction(4 if len(tokens) == 3 else 0))

    def test_empty_and_trailing_operator_are_rejected(self):
        calculator = DigitCalculator()
        with self.assertRaisesRegex(ValueError, "^EMPTY$"):
            calculator.evaluate()
        self.assertEqual(calculator.tokens, ())
        self.assertEqual(calculator.result_text, "EMPTY")
        calculator.append_number(2)
        self.assertEqual(calculator.result_text, "")
        calculator.append_operator("plus")
        with self.assertRaisesRegex(ValueError, "^INCOMPLETE$"):
            calculator.evaluate()
        self.assertEqual(calculator.tokens, (2, "plus"))
        self.assertEqual(calculator.result_text, "INCOMPLETE")
        calculator.reset()
        self.assertEqual(calculator.result_text, "")


if __name__ == "__main__":
    unittest.main()
