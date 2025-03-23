import os
import pickle

# Chemin correct basé sur l'emplacement du script
script_dir = os.path.dirname(__file__)  # Dossier du script
file_path = os.path.join(script_dir, "fraud_detector_model.pkl")

# Vérifier si le fichier existe
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Le fichier '{file_path}' est introuvable. Vérifie son emplacement.")

# Charger le modèle
with open(file_path, "rb") as f:
    model = pickle.load(f)

print("Modèle chargé avec succès !")
