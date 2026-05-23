# Alpha Expression Refinement Prompt

You are iterating on an alpha feature based on multi-agent evaluation feedback.

## Midas DSL — Available Operators (REQUIRED REFERENCE)
You MUST only use operators listed below. Do NOT invent operators (e.g. `ts_return`, `ts_quantile`, `ts_gt`, `lit` — these do NOT exist). Numeric literals are written as plain numbers (e.g. `0.5`, `60`) with no wrapper.

{{midas_dsl_skill}}

## Original Expression
{{expression}}

## Evaluation Results (JSON)
{{evaluation_result}}

## Blocking Issues
{{blocking_issues}}

## Agent Suggestions
{{suggestions}}

## Common Fixes
- High turnover          → wrap in ema() to smooth
- Short half-life        → increase lookback or add lag
- High existing corr     → residualise with cs_neutralize or sub out the correlated component
- Regime-dependent       → add if_else regime filter
- Overfit (ratio > 1.5)  → simplify — remove parameters or reduce nesting
- Weak rankIC            → try different feature combinations, NOT deeper nesting

## Hard Constraints (your refinement MUST satisfy these)
- Use ONLY operators listed in the DSL reference above. Inventing operators will cause validation failure.
- Max nesting depth: 5. Count parentheses carefully — deeply nested expressions will be rejected.
- Numeric literals are plain numbers. Do NOT wrap in `lit(...)`.
- For conditional gating, use `gt(a, b)` / `lt(a, b)` / `if_else(cond, x, y)`.
- Lookback windows must be positive integers (negative lookbacks = look-ahead bias = rejected).

## Output (YAML only)
```yaml
refined_expression: |
  the_improved_dsl_expression

changes_made:
  - "<change> — <reason>"

expected_improvement:
  - "<metric> should improve because <reason>"
```
