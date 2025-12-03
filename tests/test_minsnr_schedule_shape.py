import pytest


@pytest.mark.skip(reason="Min-SNR schedule not implemented yet; placeholder test.")
def test_minsnr_weight_shape_placeholder():
    """
    Placeholder for future Min-SNR schedule tests.

    Once Min-SNR loss weighting is implemented in ablation-harness, this test
    should be updated to:

    - construct the Min-SNR weight curve over timesteps t = 1..T
    - assert basic properties, e.g.:
        * no NaNs / infs
        * weights are within a sane numeric range
        * monotonicity or shape constraints you care about
          (e.g. emphasis on mid SNR region, de-emphasis on extreme low SNR)
    """
    # This will be replaced with real assertions once Min-SNR weights exist.
    pass