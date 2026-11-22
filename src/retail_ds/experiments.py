"""A/B test helpers used by initiatives I3, I4 and I5."""
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest


def sample_size_per_arm(baseline_rate: float, target_rate: float, alpha=0.05, power=0.8) -> int:
    effect = proportion_effectsize(target_rate, baseline_rate)
    return int(round(NormalIndPower().solve_power(effect, alpha=alpha, power=power)))


def ab_test(conv_control: int, n_control: int, conv_treat: int, n_treat: int) -> dict:
    stat, p = proportions_ztest([conv_treat, conv_control], [n_treat, n_control])
    rc, rt = conv_control / n_control, conv_treat / n_treat
    return {"control_rate": rc, "treatment_rate": rt, "relative_uplift": rt / rc - 1, "p_value": float(p)}
