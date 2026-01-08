# Install ggplot2 if needed: install.packages("ggplot2")
library(ggplot2)

# Example data: Subjects with observation times, event status (0 = censored, 1 = event)
data <- data.frame(
  subject = 1:6,                  # Subjects 1 to 6
  start = rep(0, 6),              # All start at time 0
  end = c(5, 8, 12, 15, 10, 18),  # End times
  event = c(1, 0, 1, 0, 1, 0)     # 1 = observed event, 0 = censored
)

# Create the plot
ggplot(data, aes(x = start, xend = end, y = subject, yend = subject)) +
  geom_segment(lineend = "butt", linewidth = 1.5, color = "blue") +  # Time lines (updated for linewidth)
  geom_point(data = subset(data, event == 1), aes(x = end, y = subject), 
             shape = 19, size = 4, color = "red") +  # Circles for events (size is fine here for points)
  geom_segment(data = subset(data, event == 0), 
               aes(x = end - 0.5, xend = end, y = subject, yend = subject), 
               arrow = arrow(length = unit(0.3, "cm")), color = "black") +  # Arrows for censoring
  scale_y_continuous(breaks = 1:6, labels = paste("Subjekt", 1:6)) +
  labs(title = "Prinzip der Zensierung in der Survival Analysis",
       x = "Zeit (z. B. Monate)",
       y = "Subjekte") +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5))
		
		
### Die Daten simulieren 6 Subjekte mit Startzeit 0, Endzeit und Status (event = 1 für uncensored, 0 für censored).Die Grafik zeigt, wie Zensierung die Schätzung von Überlebenskurven (z. B. via Kaplan-Meier oder Cox-Modell) beeinflusst.