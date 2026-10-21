from loan_desk import loan_math


def test_pmt_matches_known_amortization_values():
    assert round(loan_math.pmt(15000, 9.5, 36), 2) == 480.49
    assert round(loan_math.pmt(10000, 24, 24), 2) == 528.71


def test_pmt_handles_zero_rate():
    assert loan_math.pmt(12000, 0, 12) == 1000.0


def test_calculate_payment_tool_matches_pmt():
    assert loan_math.calculate_payment(15000, 9.5, 36)["monthly_payment"] == 480.49
    assert loan_math.calculate_payment(10000, 24, 24)["monthly_payment"] == 528.71


def test_calculate_payment_rejects_bad_input():
    assert "error" in loan_math.calculate_payment("not a number", 9.5, 36)
    assert "error" in loan_math.calculate_payment(1000, 9.5, 0)
    assert "error" in loan_math.calculate_payment(1000, 9.5, -12)
