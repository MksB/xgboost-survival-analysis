import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Funktion um synthetische Daten zu generieren (von dir bereitgestellt)
def generate_churn_data(n_samples=5000, seed=42):
    np.random.seed(seed)
    
    # --- 1. Feature Generierung (Die Eigenschaften der Kunden) ---
    contract_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
    monthly_charges = np.random.normal(70, 30, size=n_samples)
    monthly_charges = np.clip(monthly_charges, 20, 150)
    support_calls = np.random.poisson(1.5, size=n_samples)
    age = np.random.randint(18, 80, size=n_samples)
    random_noise = np.random.normal(0, 1, size=n_samples)

    # --- 2. Simulation der "Hazard Function" (Das Risiko) ---
    hazard_score = 0.5 
    hazard_score += (contract_type * -1.5)
    hazard_score += (monthly_charges * 0.01)
    hazard_score += (support_calls * 0.4)
    
    # --- 3. Nicht-Linearität & Interaktion ---
    interaction_mask = (monthly_charges > 100) & (support_calls > 3)
    hazard_score[interaction_mask] += 2.0 

    # --- 4. Generierung der Zeit bis zum Event (Survival Time) ---
    baseline_hazard = 0.002
    risk = baseline_hazard * np.exp(hazard_score)
    true_time_to_event = np.random.exponential(1 / risk)
    
    # --- 5. Zensierung (Censoring) simulieren ---
    observation_window = np.random.uniform(0, 730, size=n_samples)
    
    observed_time = []
    event_occurred = []
    
    for true_t, obs_t in zip(true_time_to_event, observation_window):
        if true_t < obs_t:
            observed_time.append(true_t)
            event_occurred.append(1)
        else:
            observed_time.append(obs_t)
            event_occurred.append(0)
            
    # --- 6. DataFrame erstellen ---
    df = pd.DataFrame({
        'contract_type': contract_type,
        'monthly_charges': monthly_charges,
        'support_calls': support_calls,
        'age': age,
        'random_noise': random_noise,
        'time': np.round(observed_time).astype(int),
        'event': event_occurred
    })
    
    df = df[df['time'] > 0]
    return df

# Daten generieren
df_churn = generate_churn_data(n_samples=10000)
print(f"Datensatz Größe: {df_churn.shape}")
print(f"Zensierungs-Rate: {1 - df_churn['event'].mean():.2%}")
print("\nDaten-Statistik:")
print(df_churn.describe())

# 1. Datenvorbereitung für Survival Analysis
print("\n" + "="*60)
print("1. DATENVORBEREITUNG")
print("="*60)

# Features und Zielvariablen trennen
X = df_churn.drop(['time', 'event'], axis=1)
y = df_churn[['time', 'event']].copy()

# Für XGBoost Survival brauchen wir spezielle Label-Formatierung
# Wir erstellen zwei separate Arrays für Zeit und Event
event_times = y['time'].values
event_indicators = y['event'].values

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y['event']
)

print(f"Trainingsdaten: {X_train.shape}")
print(f"Testdaten: {X_test.shape}")
print(f"Event-Rate im Training: {y_train['event'].mean():.2%}")
print(f"Event-Rate im Test: {y_test['event'].mean():.2%}")

# 2. XGBoost Survival Model mit AFT (Accelerated Failure Time)
print("\n" + "="*60)
print("2. XGBOOST SURVIVAL MODEL (AFT)")
print("="*60)

# XGBoost AFT erwartet spezielles Label-Format: 
# Für zensierte Daten: -time, für unzensierte: +time
aft_labels_train = np.where(
    y_train['event'].values == 1, 
    y_train['time'].values, 
    -y_train['time'].values
)

aft_labels_test = np.where(
    y_test['event'].values == 1,
    y_test['time'].values,
    -y_test['time'].values
)

# AFT Modell konfigurieren
dtrain = xgb.DMatrix(X_train, label=aft_labels_train)
dtest = xgb.DMatrix(X_test, label=aft_labels_test)

# Parameter für Survival Analysis (AFT)
params_aft = {
    'objective': 'survival:aft',
    'eval_metric': 'aft-nloglik',
    'aft_loss_distribution': 'normal',
    'aft_loss_distribution_scale': 1.0,
    'tree_method': 'hist',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'verbosity': 0
}

# Modell trainieren
evals_result = {}
aft_model = xgb.train(
    params_aft,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=50,
    verbose_eval=50,
    evals_result=evals_result
)

# 3. Vorhersagen und Metriken
print("\n" + "="*60)
print("3. VORHERSAGEN UND METRIKEN")
print("="*60)

# Vorhersagen für Testdaten
preds_aft = aft_model.predict(dtest)

# Für Survival Analysis: Concordance Index berechnen
def calculate_c_index(predictions, true_times, event_indicators):
    """Berechnet den Concordance Index für Survival Predictions"""
    # Höhere Predictions = höheres Risiko = kürzere Überlebenszeit
    c_index = concordance_index_censored(
        event_indicators.astype(bool),
        true_times,
        -predictions  # Negative weil höhere Predictions = höheres Risiko
    )[0]
    return c_index

c_index_aft = calculate_c_index(
    preds_aft,
    y_test['time'].values,
    y_test['event'].values
)
print(f"Concordance Index (C-Index) AFT Modell: {c_index_aft:.4f}")

# 4. Alternative: Cox Proportional Hazards mit XGBoost
print("\n" + "="*60)
print("4. XGBOOST COX PROPORTIONAL HAZARDS")
print("="*60)

# Für Cox müssen wir die Daten anders formatieren
# Wir erstellen Survival Labels für scikit-survival Format
y_train_surv = np.array(
    [(bool(e), t) for e, t in zip(y_train['event'], y_train['time'])],
    dtype=[('event', '?'), ('time', '<f8')]
)

y_test_surv = np.array(
    [(bool(e), t) for e, t in zip(y_test['event'], y_test['time'])],
    dtype=[('event', '?'), ('time', '<f8')]
)

# Baseline Cox Modell zum Vergleich
cox_model = CoxPHSurvivalAnalysis()
cox_model.fit(X_train, y_train_surv)

# C-Index für Cox Modell
cox_preds = cox_model.predict(X_test)
c_index_cox = concordance_index_censored(
    y_test['event'].values.astype(bool),
    y_test['time'].values,
    cox_preds
)[0]
print(f"C-Index Cox Modell: {c_index_cox:.4f}")

# 5. Feature Importance Analyse
print("\n" + "="*60)
print("5. FEATURE IMPORTANCE")
print("="*60)

# Feature Importance aus XGBoost Modell
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': aft_model.get_score(importance_type='gain')
}).sort_values('importance', ascending=False)

print("\nTop 10 wichtigste Features:")
print(importance_df.head(10))

# 6. Visualisierungen
print("\n" + "="*60)
print("6. VISUALISIERUNGEN")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Trainingsverlauf
axes[0, 0].plot(evals_result['train']['aft-nloglik'], label='Train', alpha=0.7)
axes[0, 0].plot(evals_result['test']['aft-nloglik'], label='Test', alpha=0.7)
axes[0, 0].set_xlabel('Boosting Rounds')
axes[0, 0].set_ylabel('Negative Log-Likelihood')
axes[0, 0].set_title('Trainingsverlauf XGBoost AFT')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Feature Importance
axes[0, 1].barh(range(len(importance_df)), importance_df['importance'])
axes[0, 1].set_yticks(range(len(importance_df)))
axes[0, 1].set_yticklabels(importance_df['feature'])
axes[0, 1].set_xlabel('Importance (Gain)')
axes[0, 1].set_title('Feature Importance')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3. Verteilung der Predictions
axes[0, 2].hist(preds_aft, bins=50, alpha=0.7, edgecolor='black')
axes[0, 2].set_xlabel('Predicted Risk Score')
axes[0, 2].set_ylabel('Count')
axes[0, 2].set_title('Verteilung der Vorhersagen')
axes[0, 2].grid(True, alpha=0.3)

# 4. Survival Time vs. Predictions
scatter = axes[1, 0].scatter(
    y_test['time'], 
    preds_aft, 
    c=y_test['event'], 
    cmap='coolwarm', 
    alpha=0.6,
    s=20
)
axes[1, 0].set_xlabel('Tatsächliche Zeit (Tage)')
axes[1, 0].set_ylabel('Predicted Risk')
axes[1, 0].set_title('Tatsächliche Zeit vs. Vorhersage')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 0], label='Event (1=Ja, 0=Nein)')

# 5. Kaplan-Meier Style Plot für Risikogruppen
# In Risikogruppen einteilen
risk_groups = pd.qcut(preds_aft, q=4, labels=['Sehr Niedrig', 'Niedrig', 'Hoch', 'Sehr Hoch'])
axes[1, 1].scatter(range(len(preds_aft)), preds_aft, c=y_test['event'], cmap='Reds', alpha=0.6)
axes[1, 1].set_xlabel('Kunden Index')
axes[1, 1].set_ylabel('Predicted Risk')
axes[1, 1].set_title('Risikoverteilung mit Events')
axes[1, 1].grid(True, alpha=0.3)

# 6. Metriken Vergleich
metrics = ['XGBoost AFT', 'Cox PH']
c_indices = [c_index_aft, c_index_cox]
bars = axes[1, 2].bar(metrics, c_indices, color=['steelblue', 'lightcoral'])
axes[1, 2].set_ylabel('C-Index')
axes[1, 2].set_title('Modellvergleich: Concordance Index')
axes[1, 2].set_ylim([0.5, 1.0])
axes[1, 2].grid(True, alpha=0.3, axis='y')

# Werte auf den Bars anzeigen
for bar, value in zip(bars, c_indices):
    height = bar.get_height()
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 7. Erweiterte Analyse: Time-dependent Predictions
print("\n" + "="*60)
print("7. ERWEITERTE ANALYSE: ZEITABHÄNGIGE VORHERSAGEN")
print("="*60)

# Wir können auch für verschiedene Zeitpunkte die Überlebenswahrscheinlichkeit vorhersagen
def predict_survival_probability(model, X_data, time_points):
    """Vorhersage der Überlebenswahrscheinlichkeit für verschiedene Zeitpunkte"""
    # Für AFT: P(T > t) = 1 - CDF der log-normalen Verteilung
    predictions = model.predict(xgb.DMatrix(X_data))
    
    # AFT Annahme: log(T) ~ Normal(μ, σ)
    # μ = prediction, σ = aft_loss_distribution_scale
    scale = params_aft['aft_loss_distribution_scale']
    
    survival_probs = {}
    for t in time_points:
        # Survival function für log-normal
        z = (np.log(t) - predictions) / scale
        survival_probs[t] = 1 - 0.5 * (1 + np.erf(z / np.sqrt(2)))
    
    return pd.DataFrame(survival_probs)

# Überlebenswahrscheinlichkeit für bestimmte Zeitpunkte
time_points = [30, 90, 180, 365, 730]  # Tage
survival_probs = predict_survival_probability(aft_model, X_test.head(10), time_points)

print("\nÜberlebenswahrscheinlichkeiten für erste 10 Kunden:")
print(survival_probs.round(3))

# 8. Modellinterpretation mit SHAP
try:
    print("\n" + "="*60)
    print("8. MODELLINTERPRETATION MIT SHAP")
    print("="*60)
    
    import shap
    
    # SHAP Explainer erstellen
    explainer = shap.TreeExplainer(aft_model)
    shap_values = explainer.shap_values(X_test)
    
    # SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("SHAP nicht installiert. Installiere mit: pip install shap")

# 9. Praktische Anwendung: Churn-Risiko Scoring
print("\n" + "="*60)
print("9. PRAKTISCHE ANWENDUNG: CHURN-RISIKO SCORING")
print("="*60)

# Top 10 Risikokunden identifizieren
risk_scores = pd.DataFrame({
    'customer_id': X_test.index[:50],
    'predicted_risk': preds_aft[:50],
    'actual_time': y_test['time'].values[:50],
    'actual_event': y_test['event'].values[:50],
    'monthly_charges': X_test['monthly_charges'].values[:50],
    'support_calls': X_test['support_calls'].values[:50],
    'contract_type': X_test['contract_type'].values[:50]
})

risk_scores = risk_scores.sort_values('predicted_risk', ascending=False)

print("\nTop 10 Risikokunden:")
print(risk_scores.head(10).to_string())

# Risikokategorien definieren
def categorize_risk(risk_score, percentiles=[0.25, 0.5, 0.75]):
    """Kategorisiert Kunden in Risikogruppen"""
    if risk_score < np.percentile(preds_aft, 25):
        return 'Niedrig'
    elif risk_score < np.percentile(preds_aft, 50):
        return 'Mittel'
    elif risk_score < np.percentile(preds_aft, 75):
        return 'Hoch'
    else:
        return 'Sehr Hoch'

risk_scores['risk_category'] = risk_scores['predicted_risk'].apply(categorize_risk)

print(f"\nRisikoverteilung in Testdaten:")
print(risk_scores['risk_category'].value_counts())


