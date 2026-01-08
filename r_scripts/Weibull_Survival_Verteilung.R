# Benötigte Packages installieren (falls nicht vorhanden) und laden
required_packages <- c("survival", "survminer", "ggplot2", "dplyr")
new_packages <- required_packages[!required_packages %in% installed.packages()[, "Package"]]
if(length(new_packages)) install.packages(new_packages)

library(survival)
library(survminer)
library(ggplot2)
library(dplyr)

# 1. Simulierte Daten aus einer Weibull-Verteilung erzeugen
set.seed(123)  # Für Reproduzierbarkeit
n <- 200
true_shape <- 2
true_scale <- 100

time <- rweibull(n, shape = true_shape, scale = true_scale)

# Zensierung simulieren (z. B. 30% rechts-zensiert)
censor_time <- runif(n, min = 50, max = 150)
observed_time <- pmin(time, censor_time)
status <- ifelse(time <= censor_time, 1, 0)  # 1 = Ereignis beobachtet, 0 = zensiert

data <- data.frame(
  time = observed_time, 
  status = status,
  event_type = factor(status, levels = c(0, 1), labels = c("Zensiert", "Ereignis"))
)

# Überblick über die Daten
cat("Datenübersicht:\n")
cat("Anzahl Beobachtungen:", n, "\n")
cat("Ereignisse:", sum(data$status), "(", round(mean(data$status)*100, 1), "%)\n")
cat("Zensierungen:", sum(1 - data$status), "(", round((1 - mean(data$status))*100, 1), "%)\n")
cat("Mittlere Beobachtungszeit:", round(mean(data$time), 2), "\n")

# 2. Weibull-Modell fitten (Accelerated Failure Time Modell)
weibull_fit <- survreg(Surv(time, status) ~ 1, data = data, dist = "weibull")

# Zusammenfassung anzeigen
cat("\n=== Weibull-Modell Zusammenfassung ===\n")
print(summary(weibull_fit))

# Geschätzte Parameter in der üblichen Weibull-Parametrisierung umrechnen
# ACHTUNG: survreg() verwendet eine andere Parametrisierung
estimated_intercept <- coef(weibull_fit)[1]  # Log-Scale Parameter
estimated_scale <- exp(estimated_intercept)  # Scale ≈ true_scale
estimated_shape <- 1 / weibull_fit$scale     # Shape ≈ true_shape

cat("\nGeschätzte Weibull-Parameter:\n")
cat("Shape (Formparameter):", round(estimated_shape, 3), 
    "(Wahrer Wert:", true_shape, ")\n")
cat("Scale (Skalenparameter):", round(estimated_scale, 3), 
    "(Wahrer Wert:", true_scale, ")\n")

# 3. Kaplan-Meier-Kurve und parametrische Weibull-Kurve plotten
# Kaplan-Meier (nicht-parametrisch)
km_fit <- survfit(Surv(time, status) ~ 1, data = data)

# Weibull-Überlebenskurve berechnen
time_seq <- seq(0, max(data$time) * 1.1, length.out = 200)
# Überlebensfunktion: S(t) = exp(-(t/scale)^shape)
weibull_survival <- exp(-(time_seq/estimated_scale)^estimated_shape)

# Erstelle Dataframe für Plot
plot_data <- data.frame(
  time = c(time_seq, summary(km_fit)$time),
  survival = c(weibull_survival, summary(km_fit)$surv),
  method = factor(c(rep("Weibull-Modell", length(time_seq)), 
                   rep("Kaplan-Meier", length(summary(km_fit)$time))),
                 levels = c("Kaplan-Meier", "Weibull-Modell"))
)

# Option 1: Kombinierter Plot mit ggplot2
cat("\n=== Erstelle Visualisierungen ===\n")

p_combined <- ggplot(plot_data, aes(x = time, y = survival, color = method, linetype = method)) +
  geom_line(data = subset(plot_data, method == "Weibull-Modell"), linewidth = 1.2) +
  geom_step(data = subset(plot_data, method == "Kaplan-Meier"), linewidth = 1.2) +
  labs(
    title = "Vergleich: Kaplan-Meier vs. Weibull-Modell",
    subtitle = paste0("n = ", n, ", ", sum(data$status), " Ereignisse (", 
                     round(mean(data$status)*100, 1), "%)"),
    x = "Zeit",
    y = "Überlebenswahrscheinlichkeit S(t)",
    color = "Methode",
    linetype = "Methode"
  ) +
  scale_color_manual(values = c("Kaplan-Meier" = "black", "Weibull-Modell" = "red")) +
  scale_linetype_manual(values = c("Kaplan-Meier" = "solid", "Weibull-Modell" = "dashed")) +
  theme_minimal() +
  theme(
    legend.position = c(0.85, 0.85),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  ) +
  ylim(0, 1) +
  geom_rug(data = data[data$status == 1, ], aes(x = time), 
           sides = "b", color = "red", alpha = 0.3, inherit.aes = FALSE) +
  geom_rug(data = data[data$status == 0, ], aes(x = time), 
           sides = "b", color = "blue", alpha = 0.3, inherit.aes = FALSE)

print(p_combined)

# Option 2: survminer Plot mit Risikotabelle
cat("\nAlternative Darstellung mit survminer...\n")
ggsurv <- ggsurvplot(
  km_fit,
  data = data,
  risk.table = TRUE,
  risk.table.height = 0.25,
  risk.table.y.text = FALSE,
  surv.median.line = "hv",
  ggtheme = theme_minimal(),
  palette = "black",
  title = "Kaplan-Meier Kurve mit Risikotabelle",
  xlab = "Zeit",
  ylab = "Überlebenswahrscheinlichkeit"
)

# Weibull-Kurve hinzufügen
ggsurv$plot <- ggsurv$plot + 
  geom_line(data = data.frame(time = time_seq, survival = weibull_survival),
            aes(x = time, y = survival, color = "Weibull-Modell"),
            linewidth = 1.2, linetype = "dashed") +
  scale_color_manual(name = "Methode", 
                     values = c("Kaplan-Meier" = "black", "Weibull-Modell" = "red")) +
  theme(legend.position = c(0.85, 0.85))

print(ggsurv)

# 4. Diagnostische Plots
cat("\n=== Diagnostische Plots ===\n")

# Weibull-Wahrscheinlichkeitsplot
par(mfrow = c(2, 2))

# Plot 1: Log-Log Überlebenskurve (sollte linear sein für Weibull-Verteilung)
plot(km_fit, fun = "cloglog", main = "Log-Log Plot: Prüfung Weibull-Annahme",
     xlab = "log(Zeit)", ylab = "log(-log(S(t)))", col = "black")
grid()
abline(a = estimated_shape * log(estimated_scale), 
       b = estimated_shape, col = "red", lwd = 2, lty = 2)
legend("topleft", legend = c("KM", "Weibull-Fit"), 
       col = c("black", "red"), lty = c(1, 2), lwd = c(1, 2))

# Plot 2: Residuen vs. angepasste Werte
residuals <- residuals(weibull_fit, type = "deviance")
plot(predict(weibull_fit), residuals, main = "Deviance-Residuen",
     xlab = "Angepasste Werte", ylab = "Deviance-Residuen")
abline(h = 0, col = "red", lty = 2)
grid()

# Plot 3: QQ-Plot der Residuen
qqnorm(residuals, main = "Normal-QQ-Plot der Residuen")
qqline(residuals, col = "red")
grid()

# Plot 4: Hazard-Funktion Vergleich
weibull_hazard <- (estimated_shape/estimated_scale) * (time_seq/estimated_scale)^(estimated_shape-1)
plot(time_seq, weibull_hazard, type = "l", col = "red", lwd = 2,
     main = "Geschätzte Hazard-Funktion",
     xlab = "Zeit", ylab = "h(t)", ylim = c(0, max(weibull_hazard)*1.1))
grid()

# Zurücksetzen der Plot-Einstellungen
par(mfrow = c(1, 1))

# 5. Modellgüte und Informationen
cat("\n=== Modellbewertung ===\n")

# Log-Likelihood und AIC
loglik <- weibull_fit$loglik[2]
aic <- AIC(weibull_fit)
cat("Log-Likelihood:", round(loglik, 3), "\n")
cat("AIC:", round(aic, 3), "\n")

# Median-Überlebenszeit
median_survival <- estimated_scale * (log(2))^(1/estimated_shape)
cat("\nGeschätzte mediane Überlebenszeit:", round(median_survival, 2), "\n")

# Überlebenswahrscheinlichkeiten zu bestimmten Zeitpunkten
time_points <- c(50, 100, 150)
surv_probs <- exp(-(time_points/estimated_scale)^estimated_shape)
cat("\nGeschätzte Überlebenswahrscheinlichkeiten:\n")
for(i in seq_along(time_points)) {
  cat("S(", time_points[i], ") = ", round(surv_probs[i], 3), "\n", sep = "")
}