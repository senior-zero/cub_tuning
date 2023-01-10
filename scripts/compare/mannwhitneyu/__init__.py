from scipy.stats import mannwhitneyu


def distributions_are_different(ref_samples, cmp_samples, alpha=0.05):
    # H0: the distributions are not different
    # H1: the distribution are different
    _, p = mannwhitneyu(ref_samples, cmp_samples)

    # Reject H0
    return p < alpha

