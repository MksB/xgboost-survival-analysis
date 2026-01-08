# Laden der benötigten Pakete
library(survival)
library(ggplot2)

# Simulation von Survival-Daten
set.seed(123)  # Für Reproduzierbarkeit
n <- 200  # Anzahl Beobachtungen
lambda <- 0.1  # Hazard-Rate (exponentielle Verteilung)
covariate <- rnorm(n)  # Ein Kovariate (z.B. Alter oder Treatment)
true_beta <- 0.5  # Wahrer Effekt des Kovariaten

# Wahre Ereigniszeiten generieren
true_time <- rexp(n, rate = lambda * exp(true_beta * covariate))

# Zensurzeiten generieren (uniform zwischen 0 und max_time)
max_time <- 20
censor_time <- runif(n, 0, max_time)

# Beobachtete Zeiten und Delta
observed_time <- pmin(true_time, censor_time)
delta <- as.numeric(true_time <= censor_time)

# Dataframe erstellen
data <- data.frame(time = observed_time, status = delta, cov = covariate)

# Kaplan-Meier-Schätzer (wahre nicht-parametrische Überlebenskurve)
km_fit <- survfit(Surv(time, status) ~ 1, data = data)

# Naïve lineare Regression (Standard-MSE, ignoriert Zensur)
lm_fit <- lm(time ~ cov, data = data)
lm_pred <- predict(lm_fit, newdata = data.frame(cov = seq(min(covariate), max(covariate), length.out = 100)))

# Cox-Modell (angepasster Loss für Survival)
cox_fit <- coxph(Surv(time, status) ~ cov, data = data)

# Überlebensvorhersagen für Cox (für einen medianen Kovariaten-Wert)
median_cov <- median(covariate)
cox_surv <- survfit(cox_fit, newdata = data.frame(cov = median_cov))

# Plot-Vorbereitung: Dataframes für Linien
# KM-Kurve
km_df <- data.frame(time = km_fit$time, surv = km_fit$surv)

# Naïve LM: Annahme einer exponentiellen Verteilung für Vorhersage (approximativ)
# (Dies ist eine Vereinfachung, um eine "Überlebenskurve" zu simulieren; in Praxis falsch)
lm_mean_time <- mean(predict(lm_fit))  # Mittlere vorhergesagte Zeit
lm_surv_times <- seq(0, max_time, length.out = 100)
lm_surv <- exp(-lm_surv_times / lm_mean_time)  # Naïve exponentielle Annahme

lm_df <- data.frame(time = lm_surv_times, surv = lm_surv)

# Cox-Kurve
cox_df <- data.frame(time = cox_surv$time, surv = cox_surv$surv)

# Grafik mit ggplot2
p <- ggplot() +
  geom_step(data = km_df, aes(x = time, y = surv, color = "Kaplan-Meier (Wahr)"), linewidth = 1) +
  geom_line(data = lm_df, aes(x = time, y = surv, color = "Naïve LM (Standard-MSE)"), linewidth = 1) +
  geom_step(data = cox_df, aes(x = time, y = surv, color = "Cox-Modell (angepasst)"), linewidth = 1) +
  scale_color_manual(values = c("Kaplan-Meier (Wahr)" = "black", "Naïve LM (Standard-MSE)" = "red", "Cox-Modell (angepasst)" = "blue")) +
  labs(title = "Vergleich: Standard-Verlust vs. Survival-Anpassung",
       subtitle = "Bias durch Ignoranz der Zensur in Standard-Methoden",
       x = "Zeit", y = "Überlebenswahrscheinlichkeit S(t)",
       color = "Methode") +
  theme_minimal() +
  ylim(0, 1)

# Plot anzeigen
print(p)

# Speichern des Plots
ggsave("survival_bias_plot.png", p, width = 8, height = 6)