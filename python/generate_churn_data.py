import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_churn_data(n_samples=5000, seed=42):
    np.random.seed(seed)
    
    # --- 1. Feature Generierung (Die Eigenschaften der Kunden) ---
    
    # Vertragsart: 0 = Monatlich (risikoreich), 1 = 1 Jahr, 2 = 2 Jahre (sehr sicher)
    contract_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
    
    # Monatliche Kosten: Normalverteilung um 70€, Standardabweichung 30
    monthly_charges = np.random.normal(70, 30, size=n_samples)
    monthly_charges = np.clip(monthly_charges, 20, 150) # Begrenzen auf realistische Werte
    
    # Anzahl Support-Anrufe: Poisson-Verteilung (meistens 0-2, selten mehr)
    support_calls = np.random.poisson(1.5, size=n_samples)
    
    # Alter des Kunden
    age = np.random.randint(18, 80, size=n_samples)
    
    # Zufälliges Rauschen (Noise), das nichts erklärt (um zu sehen, ob XGBoost es ignoriert)
    random_noise = np.random.normal(0, 1, size=n_samples)

    # --- 2. Simulation der "Hazard Function" (Das Risiko) ---
    # Wir berechnen einen Risiko-Score. Höherer Score = Schnellere Kündigung.
    
    # Basis-Risiko
    hazard_score = 0.5 
    
    # Einfluss der Features (Lineare Effekte)
    hazard_score += (contract_type * -1.5)       # Vertragslaufzeit senkt Risiko massiv
    hazard_score += (monthly_charges * 0.01)     # Höhere Kosten erhöhen Risiko leicht
    hazard_score += (support_calls * 0.4)        # Support-Probleme erhöhen Risiko stark
    
    # --- 3. Einbau von Nicht-Linearität & Interaktion (XGBoost Spezialität) ---
    # Logik: Wenn jemand viel zahlt (>100) UND oft den Support anruft (>3), 
    # ist er extrem unzufrieden -> Risiko explodiert.
    interaction_mask = (monthly_charges > 100) & (support_calls > 3)
    hazard_score[interaction_mask] += 2.0 

    # --- 4. Generierung der Zeit bis zum Event (Survival Time) ---
    # Wir nutzen eine Exponentialverteilung basierend auf dem Hazard Score.
    # T = -ln(U) / lambda * exp(score) ... vereinfacht:
    
    baseline_hazard = 0.002 # Grundwahrscheinlichkeit pro Tag
    # Risiko in Lambda umwandeln (exp macht den Score positiv)
    risk = baseline_hazard * np.exp(hazard_score)
    
    # Zeit T generieren (in Tagen)
    true_time_to_event = np.random.exponential(1 / risk)
    
    # --- 5. Zensierung (Censoring) simulieren ---
    # Die Studie endet nach 2 Jahren (730 Tage).
    # Oder Kunden sind einfach noch nicht gekündigt zum Zeitpunkt des Datenabzugs.
    
    # Wir setzen eine zufällige "Beobachtungszeit" für jeden Kunden (bis heute)
    observation_window = np.random.uniform(0, 730, size=n_samples)
    
    # Logik: Wenn das wahre Event NACH dem Beobachtungsfenster passieren würde,
    # wissen wir es noch nicht -> Zensiert.
    
    observed_time = []
    event_occurred = [] # 1 = Churn, 0 = Zensiert
    
    for true_t, obs_t in zip(true_time_to_event, observation_window):
        if true_t < obs_t:
            # Kunde hat innerhalb der Beobachtungszeit gekündigt
            observed_time.append(true_t)
            event_occurred.append(1) # Event passiert
        else:
            # Kunde ist am Ende der Beobachtungszeit noch da
            observed_time.append(obs_t)
            event_occurred.append(0) # Zensiert (wir wissen nur, er hielt bis obs_t durch)
            
    # --- 6. DataFrame erstellen ---
    df = pd.DataFrame({
        'contract_type': contract_type,
        'monthly_charges': monthly_charges,
        'support_calls': support_calls,
        'age': age,
        'random_noise': random_noise, # Feature ohne Wert
        'time': np.round(observed_time).astype(int), # Tage
        'event': event_occurred # Zielvariable
    })
    
    # Filter: Zeit muss > 0 sein
    df = df[df['time'] > 0]
    
    return df

# Daten generieren
df_churn = generate_churn_data()

# Statistik prüfen
print(f"Datensatz Größe: {df_churn.shape}")
print(f"Zensierungs-Rate: {1 - df_churn['event'].mean():.2%}") # Wie viele sind noch Kunden?
print("\nErste 5 Zeilen:")
print(df_churn.head())

# Speichern für Phase 2 (R)
# df_churn.to_csv('churn_data_synthetic.csv', index=False)
