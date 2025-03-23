import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import subprocess
import joblib
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime, timedelta

# 📂 Lancer d'autres scripts en parallèle
scripts_a_executer = [
    "scripts/data_loader.py",
    "scripts/data_preprocessing.py",
    "scripts/fraud_detection.py",
    "scripts/alerte.py"
]

processes = []
for script in scripts_a_executer:
    try:
        process = subprocess.Popen(["python", script])
        processes.append(process)
    except Exception as e:
        st.error(f"Erreur lors de l'exécution de {script}: {str(e)}")

# 📂 Charger les données CSV avec gestion des erreurs
try:
    df = pd.read_csv("transactions_nettoyees.csv")
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {str(e)}")

# 🎨 Interface graphique avec Streamlit
st.set_page_config(page_title="Dashboard Fraude Bancaire", layout="wide")
st.title("📊 Tableau de Bord des Transactions Bancaires")

# 🌍 Afficher les indicateurs clés
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Fraudes Détectées", f"{df['fraude'].sum():,}")
col3.metric("Montant Total des Fraudes", f"{df[df['fraude'] == 1]['montant'].sum():,.2f} €")

# 📌 Filtres dynamiques
date_range = st.sidebar.date_input("Sélectionnez une période", [])
montant_min, montant_max = st.sidebar.slider("Montant de la transaction", float(df["montant"].min()), float(df["montant"].max()), (float(df["montant"].min()), float(df["montant"].max())))
localisation = st.sidebar.multiselect("Localisation", df["localisation"].unique())

# Ajouter un filtre par date
if date_range:
    df_filtered = df[(df["date"] >= pd.to_datetime(date_range[0])) & (df["date"] <= pd.to_datetime(date_range[1]))]
else:
    df_filtered = df

df_filtered = df_filtered[(df_filtered["montant"] >= montant_min) & (df_filtered["montant"] <= montant_max)]
if localisation:
    df_filtered = df_filtered[df_filtered["localisation"].isin(localisation)]

# 🌍 Afficher un aperçu des données filtrées
st.subheader("Aperçu des données filtrées")
st.dataframe(df_filtered.head())

# 📈 Histogramme des montants des transactions
st.subheader("Distribution des Montants des Transactions")
fig = px.histogram(df_filtered, x="montant", nbins=30, title="Distribution des Montants")
st.plotly_chart(fig)

# 📌 Nombre de transactions frauduleuses vs normales
st.subheader("Répartition des Transactions Frauduleuses")

# Calculer les valeurs de fraude
fraude_counts = df_filtered["fraude"].value_counts().reset_index()

# Renommer les colonnes de manière explicite
fraude_counts.columns = ['fraude', 'count']

# Créer le graphique
fig = px.bar(fraude_counts, x='fraude', y='count', labels={"fraude": "Type de Transaction", "count": "Nombre"}, title="Transactions Frauduleuses vs Normales")
st.plotly_chart(fig)

# 📤 Bouton d'exportation
st.sidebar.subheader("Exporter les données")
st.sidebar.download_button(label="Télécharger CSV", data=df_filtered.to_csv(index=False), file_name="transactions_filtrees.csv", mime="text/csv")

# 🔄 Attendre la fin des autres scripts (optionnel)
for process in processes:
    process.wait()

# Charger le modèle et l'encodeur avec gestion des erreurs
try:
    model = joblib.load("fraud_detector_model.pkl")
    encoder = joblib.load("encoder.pkl")
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle ou de l'encodeur : {str(e)}")

# Vérifier si le DataFrame est vide
if df.empty:
    st.warning("⚠️ Aucune transaction trouvée !")
else:
    df.drop(columns=["id"], inplace=True, errors='ignore')
    if "localisation" in df.columns and "type_transaction" in df.columns:
        df_categorique = df[["localisation", "type_transaction"]]
        df_encoded = encoder.transform(df_categorique).toarray()
        expected_columns = encoder.get_feature_names_out()
        df_encoded = pd.DataFrame(df_encoded, columns=expected_columns)
        df.drop(columns=["localisation", "type_transaction"], inplace=True)
        df = pd.concat([df, df_encoded], axis=1)
    
    expected_features = model.feature_names_in_
    missing_features = set(expected_features) - set(df.columns)
    for col in missing_features:
        df[col] = 0
    df = df[expected_features]

    df["prediction"] = model.predict(df)

    fraude_detectee = df[df["prediction"] == 1]
    if not fraude_detectee.empty:
        st.error("🚨 Transactions suspectes détectées :")
        st.dataframe(fraude_detectee)
    else:
        st.success("✅ Aucune fraude détectée.")
