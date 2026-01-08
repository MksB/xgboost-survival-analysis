# Laden der benötigten Pakete
library(survival)
library(ggplot2)
library(dplyr)

# 1. DATEN LADEN & VORBEREITEN
# ---------------------------------------------------------
data_url <- "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
data <- read.csv(data_url)

# Daten bereinigen & Gruppennamen definieren
data$status <- ifelse(data$Churn == "Yes", 1, 0)
data$group <- factor(ifelse(data$Contract == "Month-to-month", "Monatlich (Risiko)", "Langzeit (Basis)"),
                     levels = c("Langzeit (Basis)", "Monatlich (Risiko)"))

data <- data[data$tenure > 0 & !is.na(data$group), c("tenure", "status", "group")]

# Maximale Beobachtungszeit (Tenure)
max_observed_tenure <- max(data$tenure) # Sollte 72 sein

# 2. MODELLE BERECHNEN
# ---------------------------------------------------------
km_fit <- survfit(Surv(tenure, status) ~ group, data = data)
aft_fit <- survreg(Surv(tenure, status) ~ group, data = data, dist = "weibull")

# 3. DATEN FÜR GGPLOT STRUKTURIEREN
# ---------------------------------------------------------
# ERWEITERTER PRÄDIKTIONSBEREICH: Wir gehen bis 150 Monate, um das Plateau zu zeigen
pred_probs <- seq(0.001, 0.999, by = 0.001)
surv_probs <- 1 - pred_probs

# Hilfsfunktion für Vorhersagen
get_aft_pred <- function(grp_name) {
  times <- predict(aft_fit, newdata = data.frame(group = grp_name),
                   type = "quantile", p = pred_probs)
  data.frame(time = times, surv = surv_probs,
             group = grp_name, method = "AFT Modell (Weibull)") %>%
    # Filtern, um extreme Vorhersagen jenseits von z.B. 200 Monaten zu vermeiden
    filter(time <= 200)
}

# Erstellen der AFT Datensätze
aft_long <- get_aft_pred("Langzeit (Basis)")
aft_month <- get_aft_pred("Monatlich (Risiko)")

# KM Daten extrahieren
km_data <- data.frame(
  time = km_fit$time,
  surv = km_fit$surv,
  group = rep(names(km_fit$strata), km_fit$strata)
)
km_data$group <- gsub("group=", "", km_data$group)
km_data$method <- "Kaplan-Meier (Reale Daten)"

# Alles zusammenfügen
plot_df <- rbind(aft_long, aft_month, km_data)
plot_df$group <- factor(plot_df$group, levels = c("Langzeit (Basis)", "Monatlich (Risiko)"))

# Maximale Zeit, die für den Plot benötigt wird (z.B. der 99%-Wert der Langzeitgruppe)
max_plot_time <- max(aft_long$time[aft_long$surv > 0.01])

# 4. PLOT mit optimierter X-Achse
# ---------------------------------------------------------
my_colors <- c("Langzeit (Basis)" = "#2c3e50", "Monatlich (Risiko)" = "#e74c3c")

p <- ggplot() +
  # Ebene 1: Die realen Daten (Kaplan-Meier) als Treppenstufen
  geom_step(data = subset(plot_df, method == "Kaplan-Meier (Reale Daten)"),
            aes(x = time, y = surv, color = group),
            linetype = "dotted", linewidth = 0.8, alpha = 0.7) +

  # Ebene 2: Das Modell (AFT) als glatte Linien
  geom_line(data = subset(plot_df, method == "AFT Modell (Weibull)"),
            aes(x = time, y = surv, color = group),
            linewidth = 1.2) +

  # Vertikale Linie bei maximaler beobachteter Zeit
  geom_vline(xintercept = max_observed_tenure, linetype = "dashed", color = "#666666", linewidth = 0.5) +
  # Annotation für die Linie
  annotate("text", x = max_observed_tenure + 5, y = 0.95,
           label = "Ende der beobachteten Daten", angle = 90, size = 3.5, color = "#666666") +

  # Manuelle Farb-Skala
  scale_color_manual(values = my_colors) +

  # Beschriftung und Titel
  labs(
    title = "Kundenbindungs-Analyse: Vertragsart als Churn-Treiber (mit Prognose)",
    subtitle = paste0("Vergleich der realen Daten (gepunktet, bis ", max_observed_tenure, " Monate) mit dem Weibull-Prognosemodell (Linie)."),
    x = "Laufzeit in Monaten (Tenure)",
    y = "Wahrscheinlichkeit der Kundenbindung",
    color = "Vertragssegment",
    caption = "Datenquelle: IBM Telco Churn Dataset | Modell: AFT Weibull Regression"
  ) +

  # Skalen anpassen: X-Achse erweitert und Y-Achse in Prozent
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
  # X-Achse dynamisch bis zum Prognoseende, aber gerundet auf 10er Blöcke
  scale_x_continuous(breaks = seq(0, ceiling(max_plot_time / 10) * 10, 12),
                     limits = c(0, ceiling(max_plot_time / 10) * 10)) +

  # Theme: Minimalistisch und sauber
  theme_minimal(base_size = 14) +
  theme(
    text = element_text(family = "sans", color = "#333333"),
    plot.title = element_text(face = "bold", size = 18, margin = margin(b = 10)),
    plot.subtitle = element_text(size = 12, color = "#666666", margin = margin(b = 20)),
    legend.position = "top",
    legend.justification = "left",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.line.x = element_line(color = "black")
  )

# Plot anzeigen
print(p)

# Speichern in hoher Auflösung
ggsave("telco_churn_pr_ready_optimized_x.png", p, width = 10, height = 6, dpi = 300)


#Die Grafik zeigt, dass die Kunden der Langzeit-Basis-Gruppe eine erwartete durchschnittliche Verweildauer von deutlich über 72 Monaten haben (die Kurve fällt sanft weiter ab). Die X-Achse wurde entsprechend bis zu einem Wert von ca. 108 oder 120 Monaten verlängert, um die Überlegenheit des AFT-Modells bei der Prognose von Ereignissen außerhalb des beobachteten Zeitraums zu demonstrieren.