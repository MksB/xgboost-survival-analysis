# Explorative Datenanalyse (EDA) für den synthetischen Churn-Datensatz
# Voraussetzung: Der Datensatz 'churn_data_synthetic.csv' wurde mit dem bereitgestellten Python-Code generiert.
# Installiere notwendige Pakete, falls nicht vorhanden (einmalig ausführen):
# install.packages(c("ggplot2", "survival", "corrplot", "dplyr"))

# Lade die benötigten Bibliotheken
library(ggplot2)
library(survival)
library(corrplot)
library(dplyr)

# Lade den Datensatz
data <- read.csv("churn_data_synthetic.csv")

# 1. Überblick über den Datensatz
cat("Datensatz-Größe:", nrow(data), "Zeilen,", ncol(data), "Spalten\n")
head(data)  # Erste Zeilen anzeigen
str(data)   # Struktur des Datensatzes

# Überprüfung auf fehlende Werte (sollte keine geben, da synthetisch)
any_missing <- any(is.na(data))
cat("Fehlende Werte vorhanden:", any_missing, "\n")

# 2. Zusammenfassende Statistiken
summary(data)

# Zensierungsrate (Anteil zensierter Beobachtungen, event == 0)
censor_rate <- mean(data$event == 0) * 100
cat("Zensierungsrate:", round(censor_rate, 2), "%\n")

# 3. Verteilung der Variablen
# 3.1 Kategorische Variablen
# contract_type (0: Monatlich, 1: 1 Jahr, 2: 2 Jahre)
contract_table <- table(data$contract_type)
barplot(contract_table, main="Verteilung von contract_type", xlab="Contract Type", ylab="Häufigkeit")

# support_calls (Poisson-verteilt)
support_table <- table(data$support_calls)
barplot(support_table, main="Verteilung von support_calls", xlab="Anzahl Support-Anrufe", ylab="Häufigkeit")

# event (0: Zensiert, 1: Churn)
event_table <- table(data$event)
barplot(event_table, main="Verteilung von event", xlab="Event (0: Zensiert, 1: Churn)", ylab="Häufigkeit")

# 3.2 Kontinuierliche Variablen
# monthly_charges
ggplot(data, aes(x = monthly_charges)) +
  geom_histogram(binwidth = 10, fill = "blue", color = "black") +
  labs(title = "Verteilung von monthly_charges", x = "Monatliche Kosten (€)", y = "Häufigkeit") +
  theme_minimal()

# age
ggplot(data, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "green", color = "black") +
  labs(title = "Verteilung von age", x = "Alter", y = "Häufigkeit") +
  theme_minimal()

# random_noise (sollte normalverteilt um 0 sein)
ggplot(data, aes(x = random_noise)) +
  geom_histogram(binwidth = 0.2, fill = "red", color = "black") +
  labs(title = "Verteilung von random_noise", x = "Random Noise", y = "Häufigkeit") +
  theme_minimal()

# time (Beobac	htungszeit)
ggplot(data, aes(x = time)) +
  geom_histogram(binwidth = 50, fill = "purple", color = "black") +
  labs(title = "Verteilung von time", x = "Zeit (Tage)", y = "Häufigkeit") +
  theme_minimal()

# 4. Bivariate Analysen
# 4.1 Boxplots für kontinuierliche Variablen nach event
ggplot(data, aes(x = factor(event), y = monthly_charges, fill = factor(event))) +
  geom_boxplot() +
  labs(title = "monthly_charges nach event", x = "Event", y = "Monatliche Kosten") +
  theme_minimal()

ggplot(data, aes(x = factor(event), y = support_calls, fill = factor(event))) +
  geom_boxplot() +
  labs(title = "support_calls nach event", x = "Event", y = "Support-Anrufe") +
  theme_minimal()

ggplot(data, aes(x = factor(event), y = age, fill = factor(event))) +
  geom_boxplot() +
  labs(title = "age nach event", x = "Event", y = "Alter") +
  theme_minimal()

ggplot(data, aes(x = factor(event), y = time, fill = factor(event))) +
  geom_boxplot() +
  labs(title = "time nach event", x = "Event", y = "Zeit (Tage)") +
  theme_minimal()

# 4.2 Boxplots nach contract_type
ggplot(data, aes(x = factor(contract_type), y = time, fill = factor(contract_type))) +
  geom_boxplot() +
  labs(title = "time nach contract_type", x = "Contract Type", y = "Zeit (Tage)") +
  theme_minimal()

# 4.3 Untersuchung der Interaktion (monthly_charges > 100 & support_calls > 3)
data <- data %>%
  mutate(high_cost_high_support = ifelse(monthly_charges > 100 & support_calls > 3, "Ja", "Nein"))

ggplot(data, aes(x = high_cost_high_support, y = time, fill = high_cost_high_support)) +
  geom_boxplot() +
  facet_wrap(~ event) +
  labs(title = "time nach Interaktion (high cost & high support)", x = "Interaktion", y = "Zeit (Tage)") +
  theme_minimal()

# 5. Korrelationsanalyse (für numerische Variablen)
num_vars <- data[, c("monthly_charges", "support_calls", "age", "random_noise", "time")]
corr_matrix <- cor(num_vars)
corrplot(corr_matrix, method = "circle", type = "upper", tl.cex = 0.8, title = "Korrelationsmatrix")

# 6. Survival-spezifische EDA
# Erstelle Surv-Objekt
surv_obj <- Surv(time = data$time, event = data$event)

# 6.1 Gesamte Kaplan-Meier-Kurve
km_fit <- survfit(surv_obj ~ 1)
plot(km_fit, main = "Kaplan-Meier Überlebenskurve (Gesamt)", xlab = "Zeit (Tage)", ylab = "Überlebenswahrscheinlichkeit")

# 6.2 Kaplan-Meier nach contract_type
km_contract <- survfit(surv_obj ~ data$contract_type)
plot(km_contract, col = 1:3, main = "Kaplan-Meier nach contract_type", xlab = "Zeit (Tage)", ylab = "Überlebenswahrscheinlichkeit")
legend("topright", legend = c("Monatlich (0)", "1 Jahr (1)", "2 Jahre (2)"), col = 1:3, lty = 1)

# 6.3 Kaplan-Meier nach support_calls (gruppiert, z.B. 0-1, 2-3, >3)
data$support_group <- cut(data$support_calls, breaks = c(-Inf, 1, 3, Inf), labels = c("0-1", "2-3", ">3"))
km_support <- survfit(surv_obj ~ data$support_group)
plot(km_support, col = 1:3, main = "Kaplan-Meier nach support_calls-Gruppe", xlab = "Zeit (Tage)", ylab = "Überlebenswahrscheinlichkeit")
legend("topright", legend = levels(data$support_group), col = 1:3, lty = 1)

# 6.4 Log-Rank-Test für Unterschiede (Beispiel für contract_type)
survdiff(surv_obj ~ data$contract_type)

# Dies gibt einen umfassenden Überblick über die Datenverteilungen, Beziehungen und survival-spezifische Aspekte.