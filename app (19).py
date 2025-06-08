import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prédiction des ventes", page_icon="📊", layout="centered")

# Chargement
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("📊 Application de Prédiction des ventes et quantités")

st.header("🧠 Partie 1 : Choix du type de prédiction temporelle")
granularite = st.selectbox("Niveau de granularité temporelle :", ["Jour complet", "Année + Mois", "Année"])
annee = st.selectbox("Année", list(range(2015, 2026)))
mois = st.slider("Mois", 1, 12, 6)
jour_semaine = st.slider("Jour de semaine (0=Lundi)", 0, 6, 0)

st.header("📦 Partie 2 : Type de produit")
type_produit = st.radio("Filtrer les produits par :", ["Tous les produits", "Par catégorie", "Par sous-catégorie"])

st.header("📌 Informations complémentaires")
is_holiday = st.selectbox("Jour férié ?", ["Non", "Oui"])
is_holiday_season = st.selectbox("Saison des fêtes ?", ["Non", "Oui"])
delivery_duration = st.number_input("Durée de livraison (jours)", min_value=0)

# Préparation des données
input_data = {
    "Order_Year": annee,
    "Order_Month": mois,
    "Order_DayOfWeek": jour_semaine,
    "DayOfYear": (mois - 1) * 30 + 1,
    "Is_Holiday": 1 if is_holiday == "Oui" else 0,
    "Is_Holiday_Season": 1 if is_holiday_season == "Oui" else 0,
    "Delivery_Duration": delivery_duration
}

if st.button("🧾 Prédire les Ventes et Quantités"):
    try:
        df = pd.DataFrame([input_data])
        X_scaled = scaler.transform(df)
        pred = model.predict(X_scaled)[0]

        st.success("✅ Prédictions réussies !")
        st.markdown("### Résultats de la prédiction :")
        st.markdown(f"- **Sales prédit** : 💰 **{pred:,.2f} €**")
        st.markdown(f"- **Quantité prédit** : 📦 **{int(pred * 0.08)}**")  # exemple calcul quantité
    except Exception as e:
        st.error(f"Erreur : {e}")
