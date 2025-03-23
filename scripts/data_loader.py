import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import subprocess
import joblib
from sklearn.preprocessing import OneHotEncoder

# 📂 Lancer d'autres scripts en parallèle
scripts_a_executer = [
    "scripts/data_loader.py",
    "scripts/data_preprocessing.py",
    "scripts/fraud_detection.py",
    "scripts/alerte.py"
]

processes = []
for script in scripts_a_executer:
    process = subprocess.Popen(["python", script])
    processes.append(process)

# 📂 Charger les données CSV
df = pd.read_csv("transactions_nettoyees.csv")

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

df_filtered = df[(df["montant"] >= montant_min) & (df["montant"] <= montant_max)]
if localisation:
    df_filtered = df_filtered[df_filtered["localisation"].isin(localisation)]

# 🌍 Afficher un aperçu des données
st.subheader("Aperçu des données filtrées")
st.dataframe(df_filtered.head())

# 📈 Histogramme des montants des transactions
st.subheader("Distribution des Montants des Transactions")
fig = px.histogram(df_filtered, x="montant", nbins=30, title="Distribution des Montants")
st.plotly_chart(fig)

# 📌 Nombre de transactions frauduleuses vs normales
st.subheader("Répartition des Transactions Frauduleuses")

# **Correction ici** : Convertir "fraude" en str pour éviter l'erreur de `value_counts()`
df_filtered["fraude"] = df_filtered["fraude"].astype(str)  # Convertir en chaîne pour éviter les erreurs

df_fraud_count = df_filtered["fraude"].value_counts().reset_index()
df_fraud_count.columns = ["Fraude", "Nombre"]

fig = px.bar(df_fraud_count, x="Fraude", y="Nombre", labels={"Fraude": "Type de Transaction", "Nombre": "Nombre de Transactions"}, title="Transactions Frauduleuses vs Normales")
st.plotly_chart(fig)

# 📍 Carte des Fraudes
if "latitude" in df_filtered.columns and "longitude" in df_filtered.columns:
    st.subheader("Carte des Transactions Frauduleuses")
    df_fraude = df_filtered[df_filtered["fraude"] == "1"]  # Assurez-vous que "fraude" est bien une chaîne
    fig = px.scatter_mapbox(df_fraude, lat="latitude", lon="longitude", color="montant", size="montant", hover_data=["localisation"], zoom=3, mapbox_style="open-street-map")
    st.plotly_chart(fig)

# 📤 Bouton d'exportation
st.sidebar.subheader("Exporter les données")
st.sidebar.download_button(label="Télécharger CSV", data=df_filtered.to_csv(index=False), file_name="transactions_filtrees.csv", mime="text/csv")

# 🔄 Attendre la fin des autres scripts (optionnel)
for process in processes:
    process.wait()

# Charger le modèle et l'encodeur
model = joblib.load("fraud_detector_model.pkl")
encoder = joblib.load("encoder.pkl")

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
