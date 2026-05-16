# analysis.R — Malaria Geospatial ML Ghana 260 Districts
# Spatial regression (SLM/SEM/GWR) + Poisson GLM
# Author: Valentine Golden Ghanem | ORCID: 0009-0002-8332-0220
# Usage: Rscript analysis.R
suppressPackageStartupMessages({
  library(spdep)
  library(spatialreg)
  library(MASS)
  library(dplyr)
  library(readr)
})
set.seed(42)

cat("── Loading data ──────────────────────────────────────────────────────\n")
df <- read_csv("data/processed/Ghana_Malaria_260District_MasterDataset.csv",
               show_col_types = FALSE)
cat(sprintf("Loaded: %d districts × %d variables\n", nrow(df), ncol(df)))

# ── 1. Spatial weights (KNN k=8 as per study) ────────────────────────────────
cat("\n── Building spatial weights (KNN k=8) ───────────────────────────────\n")
# Use district_id as proxy coordinates (centroid approximation)
# Replace with actual lat/lon if available in your environment
if (all(c("lat", "lon") %in% names(df))) {
  coords <- cbind(df$lon, df$lat)
} else {
  coords <- cbind(seq_len(nrow(df)), rep(0, nrow(df)))
  message("Warning: lat/lon not found — using index coords. Provide actual coordinates.")
}
knn8 <- knearneigh(coords, k = 8)
W    <- nb2listw(knn2nb(knn8), style = "W")

# ── 2. Moran's I (malaria incidence) ─────────────────────────────────────────
cat("\n── Global Moran's I ──────────────────────────────────────────────────\n")
for (var in c("incidence", "parasitemia", "xgb_risk_score")) {
  if (var %in% names(df)) {
    vals <- df[[var]]
    if (any(!is.na(vals))) {
      mi <- moran.test(vals, W, randomisation = TRUE, na.action = na.omit)
      cat(sprintf("  %-20s  I=%.4f  z=%.3f  p=%.4f\n",
                  var, mi$estimate[1], mi$statistic, mi$p.value))
    }
  }
}

# ── 3. OLS → LM tests → SLM/SEM ──────────────────────────────────────────────
cat("\n── Spatial model selection ───────────────────────────────────────────\n")
predictors <- intersect(c("itn_coverage", "water_access", "sanitation",
                           "parasitemia", "pop_density"), names(df))
fml <- as.formula(paste("incidence ~", paste(predictors, collapse = " + ")))
ols <- lm(fml, data = df)
lm_tests <- lm.RStests(ols, W, test = "all")
cat("LM diagnostics:\n"); print(lm_tests)

slm <- lagsarlm(fml, data = df, listw = W)
sem <- errorsarlm(fml, data = df, listw = W)
cat(sprintf("\n  OLS AIC=%.2f | SLM rho=%.4f AIC=%.2f | SEM lambda=%.4f AIC=%.2f\n",
            AIC(ols), slm$rho, AIC(slm), sem$lambda, AIC(sem)))

# ── 4. Negative Binomial GLM ─────────────────────────────────────────────────
cat("\n── Negative Binomial GLM (malaria incidence) ─────────────────────────\n")
nb_mod <- tryCatch(
  glm.nb(round(incidence) ~ itn_coverage + water_access + sanitation + parasitemia,
         data = df),
  error = function(e) { cat("  NB GLM failed:", conditionMessage(e), "\n"); NULL }
)
if (!is.null(nb_mod)) {
  cat(sprintf("  AIC = %.2f  theta = %.4f\n", AIC(nb_mod), nb_mod$theta))
  print(coef(summary(nb_mod)))
}

# ── 5. High-burden district characterisation ──────────────────────────────────
cat("\n── High-burden district summary ──────────────────────────────────────\n")
if ("high_burden" %in% names(df)) {
  summary_tbl <- df |>
    group_by(high_burden) |>
    summarise(across(all_of(intersect(c("incidence","parasitemia","itn_coverage",
                                        "water_access"), names(df))),
                     ~round(mean(.x, na.rm=TRUE), 2)),
              n = n(), .groups = "drop")
  print(summary_tbl)
}
cat("\nAnalysis complete.\n")
