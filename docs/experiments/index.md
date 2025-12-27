# Experiments index

### wk1

e1 – Baseline short-run sanity.
    Vanilla DDPM (current config) but short steps; confirm FID/KID pipeline still works.

e2 – Min-SNR (γ=1) short run.
    Same everything, loss.weighting = "min_snr"; log weight vs t.

e3 – Min-SNR (γ=5) short run.
    Compare loss curves + gradient stats to E1/E2.

e4 – Weight normalization ablation.
    Same as W1-E2 but normalize weights to have mean 1; check that effective LR doesn’t blow up.

e5 – Hutchinson trace probe (baseline).
    Tiny job that logs a curvature proxy (Hutchinson trace estimate) over training for vanilla.

e6 – Hutchinson trace probe (Min-SNR).
    Same as W1-E5 but with Min-SNR; compare curvature trajectories.


### wk2

e7 – Baseline reference rerun (short).
    Quick re-confirm baseline loss/curvature after any code tweaks from W1.

e8 – Min-SNR γ sweep (mini):
    Very short runs for γ ∈ {1, 3, 5} to see which looks most promising.

e9 – Scheduler interaction (linear vs cosine, very short).
    Compare Min-SNR vs vanilla under cosine schedule just to ensure no obvious pathology.

e10 – Sampler sanity (DDPM vs DDIM, fixed config).
    Small NFE (10 or 20) to confirm Min-SNR doesn’t break sampling.

e11 – KID quick check (baseline).
    5k samples, low NFE, baseline vs Min-SNR; treat as reconnaissance.

e12 – Settings playground experiment.
    One composite experiment where decide γ + schedule combo to promote into W3 full scan based on week-1/2 findings.

 
### wk3 

e13 – Baseline (linear, NFE=50, 3 seeds).

e14 – Min-SNR (linear, NFE=50, 3 seeds).

e15 – Baseline (linear, NFE=25, 3 seeds).

e16 – Min-SNR (linear, NFE=25, 3 seeds).

e17 – Baseline (cosine, NFE=25, 3 seeds).

e18 – Min-SNR (cosine, best γ, NFE=25 3 seeds).


### wk 4

e19 – Hold-out baseline (best vanilla config, 3 new seeds).
 
e20 – Hold-out Min-SNR (best γ, 3 new seeds).

e21 – Hyperparam sensitivity probe: e.g., slight LR change for Min-SNR.

e22 – NFE robustness check: run best Min-SNR config at a new NFE (e.g., 35 steps).

e23 – KID & other metric confirmation for best config, used to explain model reletive to typical others.

e24 – Tiny robustness or failure-mode experiment to test where/if Min-SNR implicitly breaks 