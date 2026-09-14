# Phase 2 Evaluation Report

## Split: train

### overall (n=70000)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.9066 | 2373.35 |
| Rct | ohm | 0.8950 | 231893 |
| Q | S*s^alpha | 0.9009 | 2.67664e-08 |
| alpha | (dimensionless) | 0.9160 | 0.0254031 |

### main (n=49000)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.9222 | 2136.17 |
| Rct | ohm | 0.9402 | 165371 |
| Q | S*s^alpha | 0.9366 | 2.17156e-08 |
| alpha | (dimensionless) | 0.9368 | 0.0221721 |

### degenerate (n=14000)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.9230 | 2197.61 |
| Rct | ohm | 0.9394 | 228054 |
| Q | S*s^alpha | 0.9178 | 2.58478e-08 |
| alpha | (dimensionless) | 0.9111 | 0.0260958 |

### low_freq_coverage (n=7000)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.7643 | 4385.15 |
| Rct | ohm | 0.4553 | 705225 |
| Q | S*s^alpha | 0.6235 | 6.39591e-08 |
| alpha | (dimensionless) | 0.7814 | 0.0466342 |

### degenerate_low_Rct_extreme (n=7092)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.9058 | 2447.84 |
| Rct | ohm | 0.9495 | 3767.09 |
| Q | S*s^alpha | 0.8595 | 3.8827e-08 |
| alpha | (dimensionless) | 0.8307 | 0.0422143 |

### degenerate_high_Rct_extreme (n=6908)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.9406 | 1940.71 |
| Rct | ohm | 0.8652 | 458315 |
| Q | S*s^alpha | 0.9772 | 1.25229e-08 |
| alpha | (dimensionless) | 0.9925 | 0.00954809 |

## Split: val

### overall (n=15000)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.8999 | 2453.14 |
| Rct | ohm | 0.8949 | 229935 |
| Q | S*s^alpha | 0.8972 | 2.74256e-08 |
| alpha | (dimensionless) | 0.9101 | 0.0262472 |

### main (n=10500)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.9160 | 2226.41 |
| Rct | ohm | 0.9467 | 165181 |
| Q | S*s^alpha | 0.9351 | 2.25649e-08 |
| alpha | (dimensionless) | 0.9319 | 0.0231377 |

### degenerate (n=3000)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.9226 | 2254.33 |
| Rct | ohm | 0.9501 | 217255 |
| Q | S*s^alpha | 0.9086 | 2.52041e-08 |
| alpha | (dimensionless) | 0.9009 | 0.0269501 |

### low_freq_coverage (n=1500)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.7409 | 4437.92 |
| Rct | ohm | 0.3575 | 708578 |
| Q | S*s^alpha | 0.6188 | 6.58937e-08 |
| alpha | (dimensionless) | 0.7750 | 0.0466085 |

### degenerate_low_Rct_extreme (n=1514)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.9083 | 2481.79 |
| Rct | ohm | 0.9562 | 3647.29 |
| Q | S*s^alpha | 0.8404 | 3.86389e-08 |
| alpha | (dimensionless) | 0.8079 | 0.0438254 |

### degenerate_high_Rct_extreme (n=1486)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.9374 | 2022.58 |
| Rct | ohm | 0.8935 | 434887 |
| Q | S*s^alpha | 0.9802 | 1.15162e-08 |
| alpha | (dimensionless) | 0.9921 | 0.00975679 |

## Split: test

### overall (n=15000)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.9058 | 2372.42 |
| Rct | ohm | 0.9072 | 221195 |
| Q | S*s^alpha | 0.8893 | 2.80358e-08 |
| alpha | (dimensionless) | 0.9135 | 0.0258334 |

### main (n=10500)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.9243 | 2115.7 |
| Rct | ohm | 0.9544 | 156089 |
| Q | S*s^alpha | 0.9239 | 2.34962e-08 |
| alpha | (dimensionless) | 0.9338 | 0.0225548 |

### degenerate (n=3000)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.9175 | 2249.22 |
| Rct | ohm | 0.9551 | 198545 |
| Q | S*s^alpha | 0.9146 | 2.49436e-08 |
| alpha | (dimensionless) | 0.9106 | 0.0263221 |

### low_freq_coverage (n=1500)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.7535 | 4415.91 |
| Rct | ohm | 0.4537 | 722238 |
| Q | S*s^alpha | 0.6024 | 6.59975e-08 |
| alpha | (dimensionless) | 0.7759 | 0.0478065 |

### degenerate_low_Rct_extreme (n=1521)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.9217 | 2294.66 |
| Rct | ohm | 0.9525 | 3680.55 |
| Q | S*s^alpha | 0.8518 | 3.83231e-08 |
| alpha | (dimensionless) | 0.8351 | 0.0428868 |

### degenerate_high_Rct_extreme (n=1479)

| Parameter | Unit | R² | MAE |
|---|---|---|---|
| Rs | ohm | 0.9131 | 2202.49 |
| Rct | ohm | 0.9005 | 398944 |
| Q | S*s^alpha | 0.9781 | 1.11842e-08 |
| alpha | (dimensionless) | 0.9928 | 0.00928689 |
