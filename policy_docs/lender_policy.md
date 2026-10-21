# Example Lender — Consumer Loan Policy

This is a fictional, prototype lending policy written for a classroom demo. It is not a
real lender's policy, and nothing here is a regulatory requirement. The numeric limits
below are kept in sync with `policy.json` at load time.

## Income verification

Every applicant's income must be confirmed with a pay stub, a bank statement, or the
most recent tax return before a loan is funded. Income stated on the application but not
backed by one of these documents counts as unverified and must not be relied on for the
affordability checks below. Overtime and bonus income can be counted only if it shows up
consistently across at least two pay stubs or two months of bank statements.

## Self-employed applicants

Self-employed applicants rarely have a pay stub. Request the last two years of signed
tax returns instead. If the two years show income declining by more than 20%, use the
lower of the two years as the applicant's income for the affordability checks, not the
average of the two.

## Payment-to-income and debt-to-income limits

The monthly payment on the loan being requested must not exceed {max_pti_pct:.0f}% of
the applicant's gross monthly income; this is the payment-to-income check. Including the
new loan's payment, the applicant's total recurring debt payments must not exceed
{max_dti_pct:.0f}% of gross monthly income; this is the debt-to-income check. Both
checks use gross income, not take-home pay after tax and deductions.

## Credit score requirements

A FICO score below {min_fico} is not an automatic decline. It means the loan needs
either collateral or a co-signer whose own FICO score is at or above {min_fico} before
it can proceed. A FICO score at or above {min_fico} with no collateral is acceptable on
its own as long as the other checks pass.

## Collateral and secured loans

For a secured loan, the loan amount may not exceed {max_ltv_pct:.0f}% of the appraised
value of the collateral; this is the loan-to-value limit. Collateral value must be
confirmed by a third-party appraisal, or for a vehicle, a standard valuation guide — the
applicant's own estimate of value is not sufficient on its own.

## Exceptions

A case that fails exactly one of the checks above, and fails it by a small margin
(within 10% of the limit), can be referred to a senior loan officer with a written
reason instead of being automatically declined. A case that fails two or more checks, or
fails a single check by more than 10%, should not be referred; it should be declined or
restructured — a smaller amount, a shorter term, or added collateral — until it passes.
