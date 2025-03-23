import pandas as pd
import mysql.connector
from sklearn.preprocessing import MinMaxScaler
import os

# Connexion à MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="transactions_db"
)

# Chargement des données
df = pd.read_sql("SELECT * FROM transactions", conn)
conn.close()

# Normalisation du montant
scaler = MinMaxScaler()
df["montant_normalisé"] = scaler.fit_transform(df[["montant"]])

# Suppression des doublons
df = df.drop_duplicates()

# Enregistrement des données nettoyées dans un fichier CSV
csv_filename = "transactions_nettoyees.csv"
df.to_csv(csv_filename, index=False)

# Affichage du chemin absolu du fichier
csv_path = os.path.abspath(csv_filename)
print(f"Fichier enregistré à : {csv_path}")
