# E11 – KID sanity check (baseline vs Min-SNR)

**ID:** E11  
**Study:** min_snr  
**Type:** Recon / metric validation  
**Primary metric:** KID (Kernel Inception Distance)  
**Secondary metric:** FID (same sampler/NFE, small n)  
**Decision use:** Confirm KID is sane + usable for later gating; not for final model claims.

---

## Question
Does our KID implementation produce finite, stable values and broadly track FID direction (baseline vs Min-SNR) under a fixed low-NFE sampling setup?

## Hypotheses
- **H1 (sanity):** KID is finite (not NaN/None) and reasonably stable across repeats (SEM not huge relative to mean).
- **H2 (direction):** At the same checkpoint and sampling setup, the ordering of runs by KID matches the ordering by FID (baseline vs Min-SNR), at least directionally.

## Design
- **Runs:** 2
  - **E11a:** baseline / constant weighting
  - **E11b:** Min-SNR (γ=5)
- **Dataset:** CIFAR-10
- **Model:** unet_cifar32, base_channels=64, EMA=0.999
- **Diffusion:** cosine β schedule
- **Training:** 10k steps, batch_size=4, AMP on
- **Sampler for eval:** DDIM
- **Eval budget:**
  - **KID:** pool n=5000, subset_size=100, repeats=20, NFE=10
  - **FID:** n=1024 (or 5000), NFE=10

## Primary comparison
E11b (Min-SNR γ=5) vs E11a (baseline)

## Success criteria
- **Must-have:** KID computed + logged with details (mean/std/sem, cache path).
- **Nice-to-have:** KID and FID agree in direction between E11a/E11b.
- **Fail mode signals:** KID unstable (huge SEM), inconsistent sign flips run-to-run, or clearly contradicts FID repeatedly on identical checkpoints.

## Planned analyses / plots 
1. Table: {run, kid_mean, kid_sem, fid, nfe, n_samples}
2. One scatter: KID vs FID across the two runs.

