# Experiments index

### wk1

W1-E1 – Baseline short-run sanity.
    Vanilla DDPM (current config) but short steps; confirm FID/KID pipeline still works.

W1-E2 – Min-SNR (γ=1) short run.
    Same everything, loss.weighting = "min_snr"; log weight vs t.

W1-E3 – Min-SNR (γ=5) short run.
    Compare loss curves + gradient stats to E1/E2.

W1-E4 – Weight normalization ablation.
    Same as W1-E2 but normalize weights to have mean 1; check that effective LR doesn’t blow up.

W1-E5 – Hutchinson trace probe (baseline).
    Tiny job that logs a curvature proxy (Hutchinson trace estimate) over training for vanilla.

W1-E6 – Hutchinson trace probe (Min-SNR).
    Same as W1-E5 but with Min-SNR; compare curvature trajectories.


### wk2

W2-E1 – Baseline reference rerun (short).
    Quick re-confirm baseline loss/curvature after any code tweaks from W1.

W2-E2 – Min-SNR γ sweep (mini):
    Very short runs for γ ∈ {1, 3, 5} to see which looks most promising.

W2-E3 – Scheduler interaction (linear vs cosine, very short).
    Compare Min-SNR vs vanilla under cosine schedule just to ensure no obvious pathology.

W2-E4 – Sampler sanity (DDPM vs DDIM, fixed config).
    Small NFE (10 or 20) to confirm Min-SNR doesn’t break sampling.

W2-E5 – FID/KID quick check (baseline).
    5k samples, low NFE, baseline vs Min-SNR; treat as reconnaissance.

W2-E6 – “Settings playground” experiment.
    One composite experiment where you decide γ + schedule combo to promote into W3 full scan based on week-1/2 findings.


### wk3 

W3-E1 – Baseline (linear, NFE=50, 3 seeds).

W3-E2 – Min-SNR (linear, NFE=50, 3 seeds).

W3-E3 – Baseline (linear, NFE=25, 3 seeds).

W3-E4 – Min-SNR (linear, NFE=25, 3 seeds).

W3-E5 – Baseline (cosine, best γ from W2, both NFEs, 3 seeds).

W3-E6 – Min-SNR (cosine, best γ, both NFEs, 3 seeds).


### wk 4

W4-E1 – Hold-out baseline (best vanilla config, 3 new seeds).

W4-E2 – Hold-out Min-SNR (best γ, 3 new seeds).

W4-E3 – Hyperparam sensitivity probe: e.g., slight LR change for Min-SNR.

W4-E4 – NFE robustness check: run best Min-SNR config at a new NFE (e.g., 35 steps).

W4-E5 – KID & other metric confirmation for best config.

W4-E6 – Tiny robustness or failure-mode experiment (e.g., starting from different init, or removing EMA) to see where Min-SNR breaks