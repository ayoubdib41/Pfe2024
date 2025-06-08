import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger les objets (à placer dans le même dossier)
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# Configuration de la page
st.set_page_config(page_title="Prédiction ventes", page_icon="📊", layout="wide")
st.title("📊 Application de Prédiction des ventes et quantités")

# Partie 1 : Granularité temporelle
st.header("🧠 Partie 1 : Choix du type de prédiction temporelle")
granularite = st.selectbox(
    "Niveau de granularité temporelle :",
    ["Année", "Année + Mois", "Jour complet"]
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    annee = st.selectbox("Année", list(range(2015, 2025)))
mois = semaine = jour = None
if granularite in ["Année + Mois", "Jour complet"]:
    with col2:
        mois = st.slider("Mois", 1, 12, 6)
if granularite == "Jour complet":
    with col3:
        semaine = st.slider("Semaine", 1, 52, 26)
    with col4:
        jour = st.slider("Jour de semaine (0=Lundi)", 0, 6, 0)

# Partie 2 : Niveau de segmentation produit
st.header("📦 Partie 2 : Type de produit")
segmentation = st.radio(
    "Filtrer les produits par :",
    ["Tous les produits", "Par catégorie", "Par sous-catégorie"]
)

# Partie 3 : Informations complémentaires
st.header("📌 Informations complémentaires")
col5, col6, col7 = st.columns(3)
with col5:
    is_holiday = st.selectbox("Jour férié ?", ["Non", "Oui"])
with col6:
    is_holiday_season = st.selectbox("Saison des fêtes ?", ["Non", "Oui"])
with col7:
    delivery_duration = st.number_input("Durée de livraison (jours)", min_value=0)

# Partie 4 : Entrée des features
st.header("📝 Remplir les caractéristiques du scénario")
input_data = {}
for feature in features:
    if feature in [
        "Order_Year", "Order_Month", "Order_Week", "Order_DayOfWeek",
        "Is_Holiday", "Is_Holiday_Season", "Delivery_Duration"
    ]:
        continue
    input_data[feature] = st.number_input(f"{feature}", value=0.0, step=1.0)

# Ajout des champs temporels et booléens
input_data.update({
    "Order_Year": annee,
    "Order_Month": mois if mois is not None else 0,
    "Order_Week": semaine if semaine is not None else 0,
    "Order_DayOfWeek": jour if jour is not None else 0,
    "Is_Holiday": 1 if is_holiday == "Oui" else 0,
    "Is_Holiday_Season": 1 if is_holiday_season == "Oui" else 0,
    "Delivery_Duration": delivery_duration
})

# Prédiction
if st.button("🔮 Prédire les Ventes et Quantités"):
    try:
        df_input = pd.DataFrame([input_data])
        X_scaled = scaler.transform(df_input)
        prediction = model.predict(X_scaled)[0]
        quantity = int(df_input.get("Quantity", [0])[0])
        st.success("✅ Prédictions réussies !")
        st.markdown("### Résultats de la prédiction :")
        st.markdown(f"- **Sales prédit** : 💰 **{prediction:.2f} €**")
        st.markdown(f"- **Quantity prédit** : 📦 **{quantity}**")
    except Exception as e:
        st.error(f"⚠️ Erreur lors de la prédiction : {e}")
