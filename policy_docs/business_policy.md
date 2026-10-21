# Example Lender — Business Loan Policy

This is a fictional, prototype business lending policy written for a classroom demo. It
is not a real lender's policy, and nothing here is a regulatory requirement. The
numeric limits below are kept in sync with `business_policy.json` at load time.

## Debt service coverage

The applicant's annual operating cash flow must cover at least {min_dscr:.2f} times
their total annual debt service — existing obligations plus the new loan being
requested. This is the debt service coverage ratio, or DSCR. A DSCR below
{min_dscr:.2f} means the company's own cash generation does not comfortably support
the debt load, even if the company is otherwise profitable on paper.

## Leverage

Total debt may not exceed {max_leverage:.1f} times annual revenue. This is a simple
proxy for leverage; it does not replace a full balance sheet review, but it catches a
company that has taken on debt far out of proportion to the size of its business.

## Minimum revenue and company age

Annual revenue must be at least {min_revenue:,.0f}, and the company must have been
operating for at least {min_years_in_business:.0f} years. A newer or smaller company is
not automatically declined, but it needs a stronger case on the other checks, or a
personal guarantee from an owner, before it can proceed.

## Public company financials

When a company is publicly traded, its revenue, total liabilities, and operating cash
flow are pulled directly from its most recent annual filing with the SEC (EDGAR), not
estimated. Total liabilities is used as a conservative proxy for total debt, since not
every filer breaks out debt-only figures the same way. A private company's figures must
be provided directly by the applicant and should be supported by financial statements.

## Scope and limitations

The risk model behind this policy is trained on synthetic data, not real company
default outcomes, and its usable range tops out well below large public companies —
a result for a very large company (above roughly two billion dollars in revenue) should
be treated as unreliable and is flagged as out of range. This policy and model are a
prototype; recalibrate before using either for a real lending decision.

## Exceptions

A case that fails exactly one of the checks above, and fails it by a small margin
(within 10% of the limit), can be referred to a senior lending officer with a written
reason instead of being automatically declined.
