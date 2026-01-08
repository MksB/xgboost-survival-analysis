# Laden der benötigten Pakete
library(survival)
library(ggplot2)

# Laden der realen Telco Churn-Daten von der URL
data_url <- "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
data <- read.csv(data_url)

# Daten vorbereiten: Relevante Spalten extrahieren und bereinigen
# Tenure als Zeit, Churn als Status (1=Churn, 0=zensiert)
data$status <- ifelse(data$Churn == "Yes", 1, 0)
data$group <- factor(ifelse(data$Contract == "Month-to-month", "Month-to-month", "Long-term"),
                     levels = c("Long-term", "Month-to-month"))  # Binäre Gruppe: Long-term (Control), Month-to-month (Treatment)

# Filtere auf valide Daten (tenure > 0, entferne NA in group)
data <- data[data$tenure > 0 & !is.na(data$group), ]
data <- data[, c("tenure", "status", "group")]  # Nur benötigte Spalten

# Kaplan-Meier-Schätzer pro Gruppe (nicht-parametrisch, zum Vergleich)
km_fit <- survfit(Surv(tenure, status) ~ group, data = data)

# AFT-Modell fitten (Weibull-Verteilung)
aft_fit <- survreg(Surv(tenure, status) ~ group, data = data, dist = "weibull")

# Zusammenfassung des Modells (optional ausgeben)
summary(aft_fit)

# Vorhersagen der Überlebenskurven aus AFT-Modell
# Für Control (group="Long-term")
pred_control <- predict(aft_fit, newdata = data.frame(group = "Long-term"), type = "quantile", p = seq(0.01, 0.99, by = 0.01))
surv_control <- 1 - seq(0.01, 0.99, by = 0.01)
time_control <- pred_control

# Für Treatment (group="Month-to-month")
pred_treatment <- predict(aft_fit, newdata = data.frame(group = "Month-to-month"), type = "quantile", p = seq(0.01, 0.99, by = 0.01))
surv_treatment <- 1 - seq(0.01, 0.99, by = 0.01)
time_treatment <- pred_treatment

# Dataframes für Plots
aft_df_control <- data.frame(time = time_control, surv = surv_control, group = "Long-term (AFT)")
aft_df_treatment <- data.frame(time = time_treatment, surv = surv_treatment, group = "Month-to-month (AFT)")

# KM-Dataframes extrahieren
km_control <- data.frame(time = km_fit[1]$time, surv = km_fit[1]$surv, group = "Long-term (KM)")
km_treatment <- data.frame(time = km_fit[2]$time, surv = km_fit[2]$surv, group = "Month-to-month (KM)")

# Kombinieren
plot_df <- rbind(aft_df_control, aft_df_treatment, km_control, km_treatment)

# Grafik mit ggplot2
p <- ggplot(plot_df, aes(x = time, y = surv, color = group, linetype = grepl("AFT", group))) +
  geom_step(linewidth = 1) +  # Step für KM, Linie für AFT (da kontinuierlich)
  scale_linetype_manual(values = c("TRUE" = "solid", "FALSE" = "dashed"), guide = "none") +
  scale_color_manual(values = c("Long-term (AFT)" = "blue", "Month-to-month (AFT)" = "red",
                                "Long-term (KM)" = "blue", "Month-to-month (KM)" = "red")) +
  labs(title = "Accelerated Failure Time (AFT) Modell: Überlebenskurven in Telco Churn",
       subtitle = "Vergleich von parametrischem AFT (Weibull) und nicht-parametrischem Kaplan-Meier\nMonth-to-month beschleunigt den Churn (verkürzt Retention-Zeit)",
       x = "Tenure (Monate)", y = "Überlebenswahrscheinlichkeit S(t) (Kein Churn)",
       color = "Gruppe und Methode") +
  theme_minimal() +
  ylim(0, 1)

# Plot anzeigen
print(p)

# Optional: Speichern des Plots
ggsave("aft_telco_churn_plot.png", p, width = 8, height = 6)