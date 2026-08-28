# Methodology and statistical assumptions

## Return convention

All inputs are periodic simple net returns greater than -1. Costs are expressed in
basis points of turnover and are deducted before compounding.

The demonstration annualizes at 252 periods per year. Callers must choose an
annualization factor that matches their sampling cadence.

## Deflated Sharpe Ratio

The implementation follows the classic Deflated Sharpe framework:

1. calculate one periodic Sharpe ratio per trial using population standard
   deviation;
2. estimate the expected maximum Sharpe under a multiple-trial null using the
   cross-trial Sharpe dispersion and the Euler-Mascheroni approximation;
3. use the selected trial's skewness and Pearson kurtosis in the probabilistic
   Sharpe variance term;
4. map the resulting test statistic through the standard normal cumulative
   distribution.

The default effective trial count equals the number of supplied trials. A caller
may provide a smaller defensible value within the observed trial count. This
public implementation deliberately does not copy the private system's
correlation estimator or frozen decision policy.

Assumptions and limitations:

- periodic returns are treated as iid;
- there is no serial-correlation or heteroskedasticity adjustment;
- the risk-free rate is periodic and defaults to zero;
- population moment estimators are used for the DSR calculation;
- trial definitions and the effective trial count remain researcher
  responsibilities.

Primary source: David H. Bailey and Marcos López de Prado,
[The Deflated Sharpe Ratio](https://doi.org/10.2139/ssrn.2460551).

## CSCV and PBO

The implementation:

1. divides the synchronized T by N return matrix into S equal contiguous
   partitions, where S is even;
2. enumerates every choice of S / 2 in-sample partitions;
3. selects the highest-Sharpe in-sample trial;
4. calculates that trial's average rank in the complementary out-of-sample
   trial scores;
5. maps relative rank omega to logit lambda = log(omega / (1 - omega));
6. estimates PBO as the share of splits with lambda less than or equal to zero.

No observation rows are truncated. A row count that is not divisible by S fails.
All-tied score sets fail because selection or ranking would be undefined.
Enumeration is bounded by an explicit maximum split count.

Primary source: David H. Bailey, Jonathan M. Borwein, Marcos López de Prado, and
Qiji Jim Zhu,
[The Probability of Backtest Overfitting](https://escholarship.org/uc/item/4w1110bb).

## Illustrative evidence gate

The command-line example checks:

- minimum observation count;
- minimum DSR probability;
- maximum PBO;
- maximum full-path drawdown.

These thresholds are demonstration inputs, not validated promotion criteria. A
passing result would permit only more research. It cannot validate alpha,
authorize execution, or establish expected profitability.

## Reproducibility

The default example fixes:

- synthetic generator seed 7;
- 512 observations;
- ten toy candidate variants;
- 2 basis points per unit of turnover, or 4 basis points for one full entry and
  exit at unit exposure;
- eight CSCV partitions and 70 complete splits.

The committed example report can be regenerated with:

    python -m hypertrader_research
