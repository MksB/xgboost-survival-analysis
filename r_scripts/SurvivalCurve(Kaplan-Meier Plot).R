# Lade notwendige Pakete
library(survival)
library(survminer)

# Simuliere Beispieldaten für 100 Kunden
set.seed(123)  # Für Reproduzierbarkeit
n <- 100
data <- data.frame(
  time = rexp(n, rate = 0.1),  # Exponentiell verteilte Zeiten (z. B. Monate bis Churn)
  status = rbinom(n, 1, 0.7),  # Status: 1 = Churn (Ereignis), 0 = Zensiert
  segment = sample(c("A", "B", "C"), n, replace = TRUE, prob = c(0.4, 0.3, 0.3))  # Kundensegmente
)

# Erhöhe Churn-Rate für Segment C, um Unterschiede zu simulieren
data$time[data$segment == "C"] <- data$time[data$segment == "C"] * 0.7  # Kürzere Zeiten für C

# Erstelle Survival-Objekt
surv_obj <- Surv(time = data$time, event = data$status)

# Fitte Kaplan-Meier-Schätzer, stratifiziert nach Segment
km_fit <- survfit(surv_obj ~ segment, data = data)

# Erstelle den Plot mit survminer (basierend auf ggplot2)
ggsurvplot(
  km_fit,
  data = data,
  conf.int = TRUE,              # Konfidenzintervalle hinzufügen
  risk.table = TRUE,            # Risikotabelle unten
  risk.table.height = 0.25,     # Höhe der Tabelle anpassen
  palette = c("blue", "green", "red"),  # Farben für Segmente A, B, C
  legend.labs = c("Segment A", "Segment B", "Segment C"),  # Legendenbeschriftung
  title = "Kaplan-Meier-Survival-Kurven für Kundensegmente",
  xlab = "Zeit (Monate)",
  ylab = "Überlebenswahrscheinlichkeit",
  ggtheme = theme_minimal()     # Minimales Theme für Klarheit
)

#Speichere den Plot als PNG
# ggsave("kaplan_meier_segments.png", width = 8, height = 6)


## Die Kurven zeigen, wie die Überlebenswahrscheinlichkeit (z. B. Wahrscheinlichkeit, dass ein Kunde bleibt) über die Zeit abnimmt. Unterschiede zwischen Segmenten können mit einem Log-Rank-Test getestet werden