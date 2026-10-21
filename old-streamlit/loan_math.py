def monthly_payment(principal, annual_rate_pct, months):
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return principal / months
    return principal * r / (1 - (1 + r) ** -months)


def all_in_rate(principal, payment, months, upfront_fees=0.0):
    """Annual rate (monthly compounding) that makes the payments equal what you really received."""
    net = principal - upfront_fees
    if net <= 0 or payment * months <= net:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        pv = payment * (1 - (1 + mid) ** -months) / mid
        lo, hi = (mid, hi) if pv > net else (lo, mid)
    return mid * 12 * 100


def summarize(principal, annual_rate_pct, months, upfront_fees=0.0, stated_payment=0.0):
    calc = monthly_payment(principal, annual_rate_pct, months)
    pay = stated_payment or calc
    total_repaid = pay * months + upfront_fees
    return {
        "monthly_payment_calculated": round(calc, 2),
        "monthly_payment_used": round(pay, 2),
        "total_repaid": round(total_repaid, 2),
        "total_cost_of_loan": round(total_repaid - principal, 2),
        "all_in_annual_rate_pct": round(all_in_rate(principal, pay, months, upfront_fees), 2),
        "payment_mismatch": bool(stated_payment) and abs(stated_payment - calc) / calc > 0.02,
    }
