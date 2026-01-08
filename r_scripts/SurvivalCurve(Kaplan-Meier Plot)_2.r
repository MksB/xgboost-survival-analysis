# Lade notwendige Pakete
library(survival)
library(survminer)

# Lade reale Telco Churn-Daten von GitHub
url <- "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
data <- read.csv(url)

# Bereite Daten vor: Time = tenure (Monate), Event = 1 если Churn == "Yes" (beobachtet), 0 sonst (zensiert)
data$time <- data$tenure
data$event <- ifelse(data$Churn == "Yes", 1, 0)

# Definiere Segmente basierend auf Contract (Month-to-month, One year, Two year)
data$segment <- data$Contract

# Entferne Zeilen mit fehlenden oder ungültigen Werten (z. B. TotalCharges kann NA sein)
data <- na.omit(data)
data <- data[data$time > 0, ]  # Ignoriere tenure = 0

# Erstelle Survival-Objekt
surv_obj <- Surv(time = data$time, event = data$event)

# Fitte Kaplan-Meier-Schätzer, stratifiziert nach Segment
km_fit <- survfit(surv_obj ~ segment, data = data)

# Erstelle den Plot mit survminer (basierend auf ggplot2)
ggsurvplot(
  km_fit,
  data = data,
  conf.int = TRUE,              # Konfidenzintervalle hinzufügen
  risk.table = TRUE,            # Risikotabelle unten
  risk.table.height = 0.25,     # Höhe der Tabelle anpassen
  palette = c("blue", "green", "red"),  # Farben für Segmente (Month-to-month, One year, Two year)
  legend.labs = c("Month-to-month", "One year", "Two year"),  # Legendenbeschriftung
  title = "Kaplan-Meier-Survival-Kurven für Telco-Kundensegmente (basierend auf Contract)",
  xlab = "Zeit (Monate)",
  ylab = "Überlebenswahrscheinlichkeit (Kundenbindung)",
  ggtheme = theme_minimal()     # Minimales Theme für Klarheit
)

# Optional: Log-Rank-Test für Signifikanz der Unterschiede zwischen Segmenten
print(survdiff(surv_obj ~ segment, data = data))

# Speichere den Plot als PNG
# ggsave("kaplan_meier_telco_segments.png", width = 8, height = 6)



##Die Kurven illustrieren, wie die Wahrscheinlichkeit der Kundenbindung über die Monate sinkt. Ein Log-Rank-Test (im Code enthalten) prüft, ob Unterschiede signifikant sind (typischerweise p < 0.001 für diesen Datensatz).