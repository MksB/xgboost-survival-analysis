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

# Optional: Speichern für Phase 2 (R)
df_churn.to_csv('churn_data_synthetic.csv', index=False)


# visuelle Überprüfung
plt.figure(figsize=(10, 6))
plt.hist(df_churn[df_churn['event']==1]['time'], bins=50, alpha=0.5, label='Gekündigt (Event=1)', color='red')
plt.hist(df_churn[df_churn['event']==0]['time'], bins=50, alpha=0.5, label='Aktiv/Zensiert (Event=0)', color='blue')
plt.legend()
plt.title("Verteilung der Ereignis-Zeiten (Synthetisch)")
plt.xlabel("Tage")
plt.ylabel("Anzahl Kunden")
plt.show()



import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

# Wir nutzen den synthetischen DataFrame 'df_churn' aus dem vorherigen Schritt

# --- SCHRITT 1: Train / Test Split ---
# Wichtig: Stratify funktioniert bei Survival-Daten nicht wie bei Klassifikation,
# da wir Event UND Zeit haben. Ein einfacher random split reicht meistens.
X = df_churn.drop(['time', 'event'], axis=1)
y = df_churn[['time', 'event']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- SCHRITT 2: Die Umwandlung (Transformation für XGBoost AFT) ---
# XGBoost 'survival:aft' benötigt Labels im Format: "Untergrenze" und "Obergrenze".
# Das ist logisch:
# - Wenn Event passiert ist: Wir wissen den genauen Tag (Untergrenze = Obergrenze).
# - Wenn Zensiert (Event=0): Wir wissen nur, der Kunde hielt bis Tag X durch.
#   Das wahre Event liegt irgendwo zwischen Tag X und Unendlich.

def convert_to_xgboost_survival_format(y_data):
    # Wir erstellen zwei Arrays
    y_lower = y_data['time'].values
    y_upper = y_data['time'].values.copy()
    
    # Wo Event = 0 (zensiert), setzen wir die Obergrenze auf Unendlich (+inf)
    # Das sagt dem Modell: "Der Ausfall passiert irgendwann nach diesem Zeitpunkt"
    censored_mask = (y_data['event'] == 0)
    y_upper[censored_mask] = np.inf
    
    return y_lower, y_upper

# Umwandlung durchführen
y_lower_train, y_upper_train = convert_to_xgboost_survival_format(y_train)
y_lower_test, y_upper_test = convert_to_xgboost_survival_format(y_test)

# --- SCHRITT 3: Erstellen der DMatrix ---
# Die DMatrix ist das interne Datenformat von XGBoost, das für Speed optimiert ist.
# Bei AFT übergeben wir die Bounds als Label.

dtrain = xgb.DMatrix(X_train)
# Setzen der Bounds (Format: label_lower_bound, label_upper_bound)
dtrain.set_float_info('label_lower_bound', y_lower_train)
dtrain.set_float_info('label_upper_bound', y_upper_train)

dtest = xgb.DMatrix(X_test)
dtest.set_float_info('label_lower_bound', y_lower_test)
dtest.set_float_info('label_upper_bound', y_upper_test)

# --- SCHRITT 4: XGBoost Hyperparameter & Training ---

params = {
    'objective': 'survival:aft',   # Accelerated Failure Time
    'eval_metric': 'aft-nloglik',  # Negative Log-Likelihood für AFT
    'aft_loss_distribution': 'normal', # Annahme über die Verteilung (normal oder extreme)
    'aft_loss_distribution_scale': 1.20, # Wichtiger Parameter bei AFT (steuert die Streuung)
    
    'tree_method': 'hist',         # Schnelleres Training für tabellarische Daten
    'learning_rate': 0.05,
    'max_depth': 4,                # Nicht zu tief, um Overfitting bei Survival zu vermeiden
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

print("Starte Training...")
model = xgb.train(
    params, 
    dtrain, 
    num_boost_round=500, 
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=50,
    verbose_eval=50
)

# --- SCHRITT 5: Evaluierung (Vorhersage & C-Index) ---

# Vorhersage: AFT sagt den log(Time) voraus, daher np.exp()
# Das Ergebnis ist die "erwartete Überlebenszeit" (Predicted Survival Time)
predicted_time = np.exp(model.predict(dtest))

# C-Index berechnen (1.0 = Perfekt, 0.5 = Zufall)
# Der C-Index prüft: Wenn Modell sagt "Kunde A lebt länger als B", stimmt das?
c_index = concordance_index(
    event_times=y_test['time'], 
    predicted_scores=predicted_time, 
    event_observed=y_test['event']
)

print("-" * 30)
print(f"Modell Performance (C-Index): {c_index:.4f}")
print("-" * 30)

# Beispiel-Output für 5 Kunden
results = X_test.head().copy()
results['Wahre_Zeit'] = y_test['time'].head()
results['Ist_Gekuendigt'] = y_test['event'].head()
results['Vorhersage_Tage'] = predicted_time[:5]

print("\nBeispiel-Vorhersagen:")
print(results[['contract_type', 'monthly_charges', 'Wahre_Zeit', 'Vorhersage_Tage']])