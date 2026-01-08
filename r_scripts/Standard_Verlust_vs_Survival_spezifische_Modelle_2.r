# Standard-Verlust vs. Survival-spezifische Modelle


# Laden der benötigten Pakete
library(survival)
library(ggplot2)

# Simulation von realistischen Telco Churn-Daten
set.seed(123)  # Für Reproduzierbarkeit
n <- 500  # Anzahl Beobachtungen (Stichprobe aus typischem Telco-Dataset)
lambda <- 0.02  # Basis-Hazard-Rate (mittlere Tenure ~50 Monate ohne Kovariaten)
monthly_charges <- rnorm(n, mean = 70, sd = 30)  # Realistische Kovariate: MonthlyCharges in USD
true_beta <- 0.01  # Wahrer Effekt: Höhere Charges erhöhen Hazard (schnellerer Churn)

# Wahre Ereigniszeiten (Churn-Zeiten) generieren
true_tenure <- rexp(n, rate = lambda * exp(true_beta * monthly_charges))

# Zensurzeiten generieren (uniform zwischen 0 und max_time)
max_time <- 72  # Maximale Beobachtungszeit in Monaten (realistisch für Telco)
censor_tenure <- runif(n, 0, max_time)

# Beobachtete Zeiten und Churn-Indikator
observed_tenure <- pmin(true_tenure, censor_tenure)
churn <- as.numeric(true_tenure <= censor_tenure)

# Dataframe erstellen
data <- data.frame(tenure = observed_tenure, churn = churn, monthly_charges = monthly_charges)

# Kaplan-Meier-Schätzer (wahre nicht-parametrische Überlebenskurve)
km_fit <- survfit(Surv(tenure, churn) ~ 1, data = data)

# Naïve lineare Regression (Standard-MSE, ignoriert Zensur)
lm_fit <- lm(tenure ~ monthly_charges, data = data)
lm_pred <- predict(lm_fit, newdata = data.frame(monthly_charges = seq(min(monthly_charges), max(monthly_charges), length.out = 100)))

# Cox-Modell (angepasster Loss für Survival)
cox_fit <- coxph(Surv(tenure, churn) ~ monthly_charges, data = data)

# Überlebensvorhersagen für Cox (für medianen MonthlyCharges-Wert)
median_charges <- median(monthly_charges)
cox_surv <- survfit(cox_fit, newdata = data.frame(monthly_charges = median_charges))

# Plot-Vorbereitung: Dataframes für Linien
# KM-Kurve
km_df <- data.frame(tenure = km_fit$time, surv = km_fit$surv)

# Naïve LM: Annahme einer exponentiellen Verteilung für Vorhersage (approximativ)
# (Dies ist eine Vereinfachung, um eine "Überlebenskurve" zu simulieren; in Praxis falsch)
lm_mean_tenure <- mean(predict(lm_fit))  # Mittlere vorhergesagte Tenure
lm_surv_times <- seq(0, max_time, length.out = 100)
lm_surv <- exp(-lm_surv_times / lm_mean_tenure)  # Naïve exponentielle Annahme
lm_df <- data.frame(tenure = lm_surv_times, surv = lm_surv)

# Cox-Kurve
cox_df <- data.frame(tenure = cox_surv$time, surv = cox_surv$surv)

# Grafik mit ggplot2
p <- ggplot() +
  geom_step(data = km_df, aes(x = tenure, y = surv, color = "Kaplan-Meier (Wahr)"), linewidth = 1) +
  geom_line(data = lm_df, aes(x = tenure, y = surv, color = "Naïve LM (Standard-MSE)"), linewidth = 1) +
  geom_step(data = cox_df, aes(x = tenure, y = surv, color = "Cox-Modell (angepasst)"), linewidth = 1) +
  scale_color_manual(values = c("Kaplan-Meier (Wahr)" = "black", "Naïve LM (Standard-MSE)" = "red", "Cox-Modell (angepasst)" = "blue")) +
  labs(title = "Vergleich: Standard-Verlust vs. Survival-Anpassung in Telco Churn",
       subtitle = "Bias durch Ignoranz der Zensur in Standard-Methoden (z.B. bei MonthlyCharges)",
       x = "Tenure (Monate)", y = "Überlebenswahrscheinlichkeit S(t) (Kein Churn)",
       color = "Methode") +
  theme_minimal() +
  ylim(0, 1)

# Plot anzeigen
print(p)

# Optional: Speichern des Plots
ggsave("telco_churn_bias_plot.png", p, width = 8, height = 6)

##############
#Die Grafik veranschaulicht eine simulierte Churn-Survival-Analyse für eine Stichprobe von n = 500 Telco-Kunden mit einer maximalen Beobachtungszeit von 72 Monaten, realistischen Churn-Raten von etwa 40–50 %, einer Zensierung von etwa 40 % und einer Kovariate monthly_charges, deren höhere Werte den Churn beschleunigen. Die Überlebensfunktion S(t) beschreibt dabei die Wahrscheinlichkeit, dass ein Kunde bis zum Zeitpunkt t nicht abwandert.


#Die Grafik zeigt eine simulierte Churn-Survival-Analyse für eine Stichprobe von 500 Telekommunikationskunden mit einer maximalen Beobachtungszeit von 72 Monaten, realistischen Churn-Raten, einer Zensierungsrate von etwa 40–50 % und der Kovariate „monthly_charges”, deren höhere Werte den Churn beschleunigen. Die Überlebensfunktion S(t) beschreibt die Wahrscheinlichkeit, dass ein Kunde bis zum Zeitpunkt t nicht abwandert.

###########################
#Realistische Anpassungen:
#
#
#n = 500: Eine Stichprobe, um Rechenzeit zu sparen (reales Dataset hat ~7000 #Beobachtungen).
#Zeit (tenure): Simuliert als Monate bis Churn (Ereignis), max_time = 72 Monate #(typischer Maximalwert in Telco-Daten).
#Hazard-Rate (lambda = 0.02): Entspricht einer mittleren Überlebenszeit von ~50 #Monaten ohne Kovariaten, realistisch für Churn-Raten um 20-30%.
#Kovariate (monthly_charges): Normalverteilt mit Mittelwert 70 USD und SD 30 USD #(basierend auf realen Telco-Daten, wo MonthlyCharges ~30-120 USD liegen).
#true_beta = 0.01: Positiver Effekt – höhere MonthlyCharges erhöhen die Hazard #(schnellerer Churn), was in realen Daten oft beobachtet wird (z. B. Korrelation von #hohen Kosten mit Abwanderung).
#Zensur: Uniform zwischen 0 und 72 Monaten, führt zu ~40-50% zensierten Fällen #(realistisch, da viele Kunden noch aktiv sind).
#Variablennamen und Labels: An Telco-Kontext angepasst (z. B. "tenure" statt "time", #"churn" statt "status", "monthly_charges" statt "cov"). Achsen und Titel auf #Churn-Survival abgestimmt.
#Überlebensinterpretation: S(t) = Wahrscheinlichkeit, dass der Kunde bis Zeit t nicht #churnt (bleibt).

#Der Code zeigt weiterhin den Bias durch Standard-MSE (naïve LM) vs. angepassten #Cox-Modell. Führen Sie ihn in R/RStudio aus (benötigt survival und ggplot2).