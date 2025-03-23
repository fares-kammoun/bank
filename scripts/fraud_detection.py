import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle
import joblib
from sklearn.preprocessing import OneHotEncoder

# 📂 Charger les données
fichier_csv = "transactions_nettoyees.csv"
df = pd.read_csv(fichier_csv)

# 🏷️ Identifier les colonnes catégorielles
categorical_cols = df.select_dtypes(include=['object']).columns

# 🎭 Convertir les colonnes catégorielles en variables numériques
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 🎯 Séparation des features (X) et du label (y)
X = df_encoded.drop(columns=["id", "fraude"])
y = df_encoded["fraude"]

# 🔀 Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ⚖️ Appliquer SMOTE pour équilibrer les classes
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 🌲 Créer et entraîner le modèle RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train_res, y_train_res)

# 🔍 Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# 🏆 Affichage des résultats
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 💾 Sauvegarde du modèle entraîné
with open("fraud_detector_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("✅ Modèle entraîné et sauvegardé sous 'fraud_detector_model.pkl' !")

# 🔎 Mettre à jour les transactions détectées comme frauduleuses dans le fichier CSV
df.loc[X_test.index, "fraude"] = y_pred  # Met à jour la colonne "fraude" dans les lignes correspondantes

# 💾 Sauvegarder les changements directement dans le fichier CSV
df.to_csv(fichier_csv, index=False)

print(f"✅ {sum(y_pred)} transactions frauduleuses détectées et mises à jour dans '{fichier_csv}' !")

# 🎭 Enregistrer l'encodeur pour les variables catégoriques
encoder = OneHotEncoder(handle_unknown="ignore")
encoder.fit(df[["localisation", "type_transaction"]])  # Utiliser le dataframe d'entraînement
joblib.dump(encoder, "encoder.pkl")

print("✅ Encodeur sauvegardé avec succès !")
