# Laden der benötigten Pakete
library(survival)
library(ggplot2)

# Simulation von AFT-Daten
set.seed(456)  # Für Reproduzierbarkeit
n <- 300  # Anzahl Beobachtungen
group <- factor(rbinom(n, 1, 0.5), labels = c("Control", "Treatment"))  # Kovariate: 0=Control, 1=Treatment
true_beta <- -0.5  # Negativer Effekt: Treatment verkürzt die Zeit (Beschleunigung)
scale <- 1  # Skalierung für Weibull

# Wahre Ereigniszeiten generieren (Weibull-AFT: log(T) = beta * group + epsilon)
# Für Weibull: epsilon ~ extreme value distribution
true_log_time <- true_beta * as.numeric(group) + rweibull(n, shape = 1/scale, scale = 1)
true_time <- exp(true_log_time)

# Zensurzeiten generieren (uniform)
max_time <- 10
censor_time <- runif(n, 0, max_time)

# Beobachtete Zeiten und Status
observed_time <- pmin(true_time, censor_time)
status <- as.numeric(true_time <= censor_time)

# Dataframe erstellen
data <- data.frame(time = observed_time, status = status, group = group)

# Kaplan-Meier-Schätzer pro Gruppe (nicht-parametrisch, zum Vergleich)
km_fit <- survfit(Surv(time, status) ~ group, data = data)

# AFT-Modell fitten (Weibull-Verteilung)
aft_fit <- survreg(Surv(time, status) ~ group, data = data, dist = "weibull")

# Zusammenfassung des Modells (optional ausgeben)
summary(aft_fit)

# Vorhersagen der Überlebenskurven aus AFT-Modell
# Für Control (group=0)
pred_control <- predict(aft_fit, newdata = data.frame(group = "Control"), type = "quantile", p = seq(0.01, 0.99, by = 0.01))
surv_control <- 1 - seq(0.01, 0.99, by = 0.01)
time_control <- pred_control

# Für Treatment (group=1)
pred_treatment <- predict(aft_fit, newdata = data.frame(group = "Treatment"), type = "quantile", p = seq(0.01, 0.99, by = 0.01))
surv_treatment <- 1 - seq(0.01, 0.99, by = 0.01)
time_treatment <- pred_treatment

# Dataframes für Plots
aft_df_control <- data.frame(time = time_control, surv = surv_control, group = "Control (AFT)")
aft_df_treatment <- data.frame(time = time_treatment, surv = surv_treatment, group = "Treatment (AFT)")

# KM-Dataframes extrahieren
km_control <- data.frame(time = km_fit[1]$time, surv = km_fit[1]$surv, group = "Control (KM)")
km_treatment <- data.frame(time = km_fit[2]$time, surv = km_fit[2]$surv, group = "Treatment (KM)")

# Kombinieren
plot_df <- rbind(aft_df_control, aft_df_treatment, km_control, km_treatment)

# Grafik mit ggplot2
p <- ggplot(plot_df, aes(x = time, y = surv, color = group, linetype = grepl("AFT", group))) +
  geom_step(linewidth = 1) +  # Step für KM, Linie für AFT (da kontinuierlich)
  scale_linetype_manual(values = c("TRUE" = "solid", "FALSE" = "dashed"), guide = "none") +
  scale_color_manual(values = c("Control (AFT)" = "blue", "Treatment (AFT)" = "red", 
                                "Control (KM)" = "blue", "Treatment (KM)" = "red")) +
  labs(title = "Accelerated Failure Time (AFT) Modell: Überlebenskurven",
       subtitle = "Vergleich von parametrischem AFT (Weibull) und nicht-parametrischem Kaplan-Meier\nTreatment beschleunigt das Ereignis (verkürzt Überlebenszeit)",
       x = "Zeit", y = "Überlebenswahrscheinlichkeit S(t)",
       color = "Gruppe und Methode") +
  theme_minimal() +
  ylim(0, 1)

# Plot anzeigen
print(p)

# Optional: Speichern des Plots
ggsave("aft_model_plot.png", p, width = 8, height = 6)


###Der Plot vergleicht die AFT-vorhergesagten Überlebenskurven (durchgezogene Linien) mit Kaplan-Meier (gestrichelt). Der Treatment-Effekt zeigt eine schnellere Abnahme der Überlebenswahrscheinlichkeit, was die "Beschleunigung" illustriert. Blaue Kurven: Control (längere Zeiten); Rote Kurven: Treatment (verkürzte Zeiten).