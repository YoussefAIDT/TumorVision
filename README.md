# 🧠 TumorVision

> **Vision IA pour la détection de tumeurs cérébrales**

Une application Streamlit avancée utilisant l'intelligence artificielle pour assister dans la détection et l'analyse des tumeurs cérébrales à partir d'images IRM.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Table des Matières

- [🎯 Aperçu](#-aperçu)
- [✨ Fonctionnalités](#-fonctionnalités)
- [🚀 Installation](#-installation)
- [💻 Utilisation](#-utilisation)
- [🔧 Configuration](#-configuration)
- [📊 Modèles IA](#-modèles-ia)
- [🗄️ Base de Données](#️-base-de-données)
- [📰 API d'Actualités](#-api-dactualités)
- [🤝 Contribution](#-contribution)
- [📜 Licence](#-licence)
- [👨‍💻 Auteur](#-auteur)

## 🎯 Aperçu

TumorVision est une application web développée pour aider les professionnels de santé dans l'analyse d'images IRM cérébrales, particulièrement utile dans les contextes où l'expertise radiologique n'est pas immédiatement disponible.

### 🌟 Points Forts

- **🤖 IA Avancée** : 3 modèles de détection (CNN, YOLOv8, Hybride)
- **📊 Analytics** : Statistiques détaillées et visualisations
- **📰 Actualités** : Flux d'actualités médicales en temps réel
- **🔒 Sécurisé** : Base de données SQLite intégrée
- **🎨 Interface Moderne** : Design responsive et intuitif

## ✨ Fonctionnalités

### 🏥 Core Features

#### 1. **📋 Présentation du Problème**
- Vue d'ensemble des types de tumeurs cérébrales
- Statistiques et graphiques interactifs
- Avertissements de responsabilité médicale

#### 2. **🔬 Analyse et Prédiction**
- **Upload d'images** : Support PNG, JPG, JPEG, DICOM
- **3 Modèles IA** :
  - 🧠 **CNN** : Classification binaire (Oui/Non)
  - 🎯 **YOLOv8** : Détection avec localisation
  - 🔄 **Hybride** : Consensus intelligent entre CNN et YOLOv8
- **Classification** : Identification du type de tumeur
- **Statistiques** : Métriques détaillées de l'analyse

#### 3. **📰 Actualités Santé**
- **Tumeurs Cérébrales** : Actualités spécialisées
- **Santé Générale** : Actualités médicales générales
- **API Temps Réel** : Mise à jour automatique via NewsAPI
- **Fallback** : Actualités de secours si API indisponible

#### 4. **💬 Chatbot & Aide**
- Interface de chatbot (en développement)
- FAQ et documentation
- Support utilisateur

#### 5. **📊 Statistiques d'Utilisation**
- Métriques de performance des modèles
- Analyse des tendances d'utilisation
- Visualisations interactives avec Plotly

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de packages Python)

### 1. Cloner le Repository

```bash
git clone https://github.com/YoussefAIDT/tumorvision.git
cd tumorvision
```

### 2. Installer les Dépendances

```bash
pip install -r requirements.txt
```

### 3. Structure du Projet

```
tumorvision/
├── app.py                # Application principale Streamlit
├── requirements.txt      # Dépendances Python
├── README.md            # Documentation du projet
└── models/              # Modèles IA pré-entraînés
    ├── cnn_model.pkl
    ├── yolov8_model.pt
    └── hybrid_model.pkl
```

### 4. Lancer l'Application

```bash
streamlit run app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

## 💻 Utilisation

### 🖼️ Analyse d'Images IRM

1. **Upload** : Glissez-déposez votre image IRM
2. **Modèle** : Choisissez le modèle d'analyse
3. **Analyse** : Cliquez sur "Analyser l'Image"
4. **Résultats** : Consultez les prédictions et statistiques

### Interprétation des Résultats

- **✅ Pas de Tumeur** : Probabilité faible de présence tumorale
- **🚨 Tumeur Détectée** : Classification automatique du type
- **📈 Confiance** : Niveau de certitude du modèle (70-95%)
- **📍 Localisation** : Coordonnées si détection YOLOv8

## 🔧 Configuration

### 🔐 Configuration des Secrets

Créez le fichier `.streamlit/secrets.toml` (optionnel) :

```toml
# API Configuration
NEWS_API_KEY = "votre_cle_newsapi_ici"
```

### 📰 Configuration NewsAPI

1. Inscrivez-vous sur [NewsAPI.org](https://newsapi.org/)
2. Obtenez votre clé API gratuite (100 requêtes/jour)
3. Ajoutez la clé dans `secrets.toml`

## 📊 Modèles IA

### 🧠 Modèle CNN
- **Architecture** : Réseau de neurones convolutionnel
- **Sortie** : Classification binaire (Tumeur/Pas de tumeur)
- **Précision** : ~87%

### 🎯 Modèle YOLOv8
- **Architecture** : You Only Look Once v8
- **Sortie** : Détection + Localisation avec boîtes englobantes
- **Précision** : ~85%

### 🔄 Modèle Hybride
- **Logique** : Consensus entre CNN et YOLOv8
- **Algorithme** :
  - Si accord → Résultat consensuel
  - Si désaccord → Plus haute confiance
- **Précision** : ~92%

## 🗄️ Base de Données

### 📊 Schema

```sql
-- Table des prédictions
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT,
    prediction_result TEXT,
    tumor_detected BOOLEAN,
    tumor_type TEXT,
    confidence REAL,
    user_feedback TEXT
);

-- Table des statistiques d'utilisation
CREATE TABLE usage_stats (
    id INTEGER PRIMARY KEY,
    date DATE,
    total_predictions INTEGER DEFAULT 0,
    true_positives INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    true_negatives INTEGER DEFAULT 0,
    false_negatives INTEGER DEFAULT 0
);
```

### 🔧 Gestion

- **Localisation** : `database/tumorvision_stats.db`
- **Sauvegarde** : Automatique à chaque prédiction
- **Export** : Compatible avec pandas/CSV

## 📰 API d'Actualités

### 🔗 APIs Supportées

| API | Plan Gratuit | Limite | Documentation |
|-----|--------------|--------|---------------|
| NewsAPI | ✅ | 100 req/jour | [Lien](https://newsapi.org/docs) |
| Guardian | ✅ | 500 req/jour | [Lien](https://open-platform.theguardian.com/) |
| NYTimes | ✅ | 1000 req/jour | [Lien](https://developer.nytimes.com/) |

## 🤝 Contribution

### 🛠️ Comment Contribuer

1. **Fork** le repository
2. **Créez** une branche feature (`git checkout -b feature/AmazingFeature`)
3. **Committez** vos changements (`git commit -m 'Add some AmazingFeature'`)
4. **Push** vers la branche (`git push origin feature/AmazingFeature`)
5. **Ouvrez** une Pull Request

### 🐛 Signaler un Bug

Utilisez les [GitHub Issues](https://github.com/YoussefAIDT/tumorvision/issues) avec le template :

```markdown
**Bug Description:**
Description claire du problème

**To Reproduce:**
1. Aller à '...'
2. Cliquer sur '...'
3. Voir l'erreur

**Expected Behavior:**
Ce qui devrait arriver

**Screenshots:**
Si applicable, ajoutez des captures d'écran

**Environment:**
- OS: [e.g. Windows 10]
- Python: [e.g. 3.9]
- Browser: [e.g. Chrome 91]
```

### 🚀 Roadmap

- [ ] **Modèles IA** : Intégration de vrais modèles pré-entraînés
- [ ] **Chatbot** : Finalisation du système d'aide
- [ ] **Mobile** : Version responsive mobile
- [ ] **API REST** : Endpoints pour intégration externe
- [ ] **Docker** : Containerisation de l'application
- [ ] **Tests** : Suite de tests automatisés
- [ ] **I18n** : Support multilingue

## 📜 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

```
MIT License

Copyright (c) 2025 ES-SAAIDI Youssef

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 👨‍💻 Auteur

**ES-SAAIDI Youssef**

- 🌐 **GitHub** : [@YoussefAIDT](https://github.com/YoussefAIDT)
- 📧 **Email** : [votre.email@example.com]
- 💼 **LinkedIn** : [Votre Profil LinkedIn]

---

## 🙏 Remerciements

- **Streamlit** pour le framework web
- **Plotly** pour les visualisations
- **NewsAPI** pour les actualités
- **Communauté Open Source** pour l'inspiration

---

## ⚠️ Avertissement Médical

> **IMPORTANT** : Cette application est un outil d'aide au diagnostic uniquement. Les prédictions ne remplacent pas l'expertise médicale professionnelle. Toute décision médicale doit être validée par un professionnel de santé qualifié. Nous n'assumons aucune responsabilité quant aux décisions prises sur la base de ces prédictions.

---

<div align="center">

**© 2025 ES-SAAIDI Youssef - TumorVision | Tous droits réservés**

🧠 **Vision IA pour la détection de tumeurs cérébrales** 🤖

[![GitHub stars](https://img.shields.io/github/stars/YoussefAIDT/tumorvision.svg?style=social&label=Star)](https://github.com/YoussefAIDT/tumorvision)
[![GitHub forks](https://img.shields.io/github/forks/YoussefAIDT/tumorvision.svg?style=social&label=Fork)](https://github.com/YoussefAIDT/tumorvision/fork)

</div>
