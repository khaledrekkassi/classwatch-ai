# 🎓 ClassWatch AI - Système de Monitoring d'Attention en Classe

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![YOLO](https://img.shields.io/badge/YOLO-v8-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Description

**ClassWatch AI** est un système intelligent de monitoring d'attention en classe utilisant la vision par ordinateur et l'intelligence artificielle.

### ✨ Fonctionnalités

- 🔍 **Détection YOLO v8** : Personnes et téléphones
- 👤 **Reconnaissance faciale** : MediaPipe
- ⏱️ **Tracking continu** : Suivi des étudiants frame par frame
- 📱 **Détection téléphones** : Utilisation non autorisée
- 🗣️ **Détection conversations** : Entre étudiants
- 📏 **Détection proximité** : Trop proche (>5s)
- 📸 **Captures automatiques** : 5 catégories (new_student, orange, red, conversation, proximity)
- 🤖 **LLM intégré** : Google Gemini / Groq pour rapports
- 🎨 **Interface moderne** : Dashboard temps réel
- 📊 **Système RAG** : Analyse comportementale multi-dossiers

## 🚀 Installation Rapide

### Prérequis

- Python 3.9+
- Webcam
- 2 GB d'espace disque

### Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-username/classwatch-ai.git
cd classwatch-ai

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API

# 4. Lancer l'application web
python web_app.py

# 5. Accéder à l'interface
# http://localhost:5000
```

## 📁 Structure du Projet

```
classwatch-ai/
├── web_app.py              # Application Flask principale
├── rag_professionnel.py    # Système RAG d'analyse
├── index.html              # Interface web
├── requirements.txt        # Dépendances Python
├── .env.example           # Configuration exemple
└── README.md              # Ce fichier
```

## 🔧 Configuration

### Variables d'environnement (.env)

```bash
# Google Gemini (Recommandé - 3M tokens/jour gratuits)
GOOGLE_API_KEY=votre_clé_ici

# Groq (Fallback - 500K tokens/jour)
GROQ_API_KEY=votre_clé_ici
```

**Obtenir les clés :**
- Google Gemini : https://aistudio.google.com/app/apikey
- Groq : https://console.groq.com

## 📊 Utilisation

### 1. Application Web (Monitoring temps réel)

```bash
python web_app.py
```

Ouvrez http://localhost:5000 dans votre navigateur.

**Fonctionnalités disponibles :**
- ✅ Flux vidéo en direct
- ✅ Statistiques temps réel
- ✅ Liste des étudiants avec détails
- ✅ Captures automatiques
- ✅ Chat avec assistant IA
- ✅ Génération de rapports

### 2. Système RAG (Analyse comportementale)

```bash
python rag_professionnel.py
```

Analyse les captures d'écran dans les dossiers :
- `screenshots/conversation/`
- `screenshots/red_distraction/`
- `screenshots/orange_distraction/`

Génère un rapport HTML professionnel avec métriques détaillées.

## 🎯 Captures Automatiques

Le système prend automatiquement des captures dans ces situations :

1. **new_student** : Nouveau visage détecté
2. **orange_distraction** : Distraction 10-30s
3. **red_distraction** : Distraction >30s
4. **conversation** : Conversation détectée
5. **proximity** : Proximité excessive >5s

Les captures sont sauvegardées avec métadonnées JSON complètes.

## 🤖 Assistant IA

L'assistant utilise Google Gemini ou Groq pour :

- 📋 Générer des rapports de classe complets
- 💬 Répondre à vos questions sur la classe
- 📊 Analyser les tendances d'attention
- 🎯 Donner des recommandations pédagogiques

**Exemples de questions :**
- "Quels sont les étudiants les plus distraits ?"
- "Analyse les tendances d'attention"
- "Donne des recommandations pour cette classe"

## 📸 Renommer un Étudiant

Double-cliquez sur le nom dans la liste pour renommer. Le nom est sauvegardé de manière permanente dans `students_database.json`.

## 🔒 Sécurité et Confidentialité

- ✅ **Données locales** : Tout est stocké localement
- ✅ **Pas de cloud** : Sauf LLM optionnel (Gemini/Groq)
- ✅ **Embeddings uniquement** : Pas de photos stockées
- ✅ **Base chiffrée** : Format JSON sécurisé

## 🐛 Dépannage

### Caméra non détectée

```python
# Dans web_app.py, ligne ~250
selected_camera_index = 0  # Essayer 1, 2, etc.
```

### Erreur LLM

Vérifiez vos clés API dans le fichier `.env`.

### Performance lente

- Réduire la résolution vidéo
- Désactiver la détection de posture
- Augmenter `decay_rate` dans le tracker

## 📦 Dépendances Principales

- **Flask** 3.0+ : Serveur web
- **OpenCV** 4.8+ : Traitement vidéo
- **YOLO v8** : Détection d'objets
- **MediaPipe** : Reconnaissance faciale
- **NumPy** : Calculs numériques
- **Google Gemini API** : LLM gratuit

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📜 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👨‍💻 Auteur

**Ali** - Développeur IA & Vision par Ordinateur

## 🙏 Remerciements

- **Ultralytics** pour YOLO v8
- **Google** pour MediaPipe et Gemini API
- **OpenCV** pour le traitement vidéo
- **Flask** pour le framework web

---

⭐ **Si ce projet vous aide, pensez à mettre une étoile !** ⭐
