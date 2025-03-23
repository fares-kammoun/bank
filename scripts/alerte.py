import pandas as pd
import joblib
import mysql.connector
from sklearn.preprocessing import OneHotEncoder

# Connexion à MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="transactions_db"
)
cursor = conn.cursor()

# Charger le modèle et l'encodeur
model = joblib.load("fraud_detector_model.pkl")  # Charge le modèle RandomForest
encoder = joblib.load("encoder.pkl")  # Charge l'encodeur utilisé à l'entraînement

# Récupérer les dernières transactions
df = pd.read_sql("SELECT * FROM transactions ORDER BY id DESC LIMIT 100", conn)

# Vérifier si le DataFrame est vide
if df.empty:
    print("⚠️ Aucune transaction trouvée !")
    exit()

# Suppression des colonnes inutiles
df.drop(columns=["id"], inplace=True, errors='ignore')

# Vérifier si l'encodeur est bien un OneHotEncoder
if not isinstance(encoder, OneHotEncoder):
    raise TypeError("❌ Erreur : 'encoder.pkl' n'est pas un OneHotEncoder valide !")

# Vérifier que toutes les colonnes catégoriques existent
if "localisation" in df.columns and "type_transaction" in df.columns:
    df_categorique = df[["localisation", "type_transaction"]]

    # Transformer les valeurs catégoriques
    df_encoded = encoder.transform(df_categorique).toarray()  # Convertir en array
    expected_columns = encoder.get_feature_names_out()

    # Vérifier la forme des données
    print(f"✅ Encodage réussi : {df_encoded.shape[1]} colonnes générées, {len(expected_columns)} attendues.")

    # Création du DataFrame encodé avec gestion des erreurs
    if df_encoded.shape[1] != len(expected_columns):
        raise ValueError(f"❌ Nombre de colonnes encodées incorrect : {df_encoded.shape[1]} vs {len(expected_columns)}.")

    df_encoded = pd.DataFrame(df_encoded, columns=expected_columns)  

    # Supprimer les colonnes originales et ajouter les colonnes encodées
    df.drop(columns=["localisation", "type_transaction"], inplace=True)
    df = pd.concat([df, df_encoded], axis=1)  # Fusion des colonnes encodées

# Vérifier les colonnes attendues par le modèle
expected_features = model.feature_names_in_

# Ajustement des colonnes pour correspondre exactement au modèle
missing_features = set(expected_features) - set(df.columns)
extra_features = set(df.columns) - set(expected_features)

# Ajouter les colonnes manquantes avec des zéros
for col in missing_features:
    df[col] = 0

# Supprimer les colonnes en trop
df = df[expected_features]

# Prédiction
df["prediction"] = model.predict(df)

# Affichage des alertes de fraude
fraude_detectee = df[df["prediction"] == 1]
if not fraude_detectee.empty:
    print("🚨 Transactions suspectes détectées :")
    print(fraude_detectee)
else:
    print("✅ Aucune fraude détectée.")

# Fermeture de la connexion
conn.close()
