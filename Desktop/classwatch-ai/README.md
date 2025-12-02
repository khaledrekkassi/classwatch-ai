# 🎓 ClassWatch AI  
### Système Avancé de Monitoring d’Attention en Environnement Pédagogique  
**Hackathon LLM – 24 décembre 2025**  
**Équipe : Khalid Rekkassi · Ali Houaoui · Youcef Belhadef · Bilel Elkeddari**

---

## 1. 📘 Introduction

ClassWatch AI est une solution intégrée de monitoring d’attention conçue pour les environnements pédagogiques.  
Elle combine la vision par ordinateur, l’analyse comportementale et les modèles de langage avancés (LLM) afin de fournir une évaluation en temps réel de l’engagement des étudiants.

Le projet a été développé dans le cadre du **Hackathon LLM 2025**, avec pour objectif de démontrer l’efficacité de l’IA dans l’amélioration de la qualité d’enseignement et du pilotage pédagogique.

---

## 2. 🎯 Objectifs du Projet

- Fournir un système automatisé permettant de mesurer l’attention des étudiants.  
- Identifier les comportements non conformes (distraction, conversations, usage du téléphone).  
- Offrir un tableau de bord temps réel.  
- Générer des rapports exploitables par les enseignants.  
- Illustrer une intégration multimodale complète (vision + NLP + RAG).

---

## 3. 🏗️ Architecture Fonctionnelle

La solution repose sur trois modules principaux :

### 3.1. 👁️ Vision par Ordinateur
- Détection d’objets via **YOLO v8** (personnes, téléphones).  
- Reconnaissance faciale via **MediaPipe**.  
- Suivi persistant des étudiants.  
- Détection d’événements : distraction, conversation, proximité.

### 3.2. 🧠 Analyse IA & Comportement
- Classification des événements (new_student, orange, red, conversation, proximity).  
- Captures contextualisées automatiques.  
- Archivage structuré avec métadonnées.

### 3.3. 💬 Intelligence Artificielle Conversationnelle
- Intégration des API **Google Gemini** et **Groq**.  
- Analyse, synthèse, génération de rapports.  
- Module **RAG** pour analyses multi-dossiers.

---

## 4. ⭐ Caractéristiques Clés

- Détection et tracking en temps réel.  
- Statistiques instantanées d’engagement.  
- Interface web ergonomique.  
- Rapports générés automatiquement.  
- Analyse comportementale sur plusieurs sources.  
- Respect strict de la confidentialité (traitement local).

---

## 5. 🛠️ Installation

### 5.1. Prérequis
- Python 3.9+  
- Webcam  
- Linux / Windows / macOS  
- Clés API Gemini / Groq (optionnelles)

### 5.2. Procédure

```bash
git clone https://github.com/votre-username/classwatch-ai.git
cd classwatch-ai

pip install -r requirements.txt

cp .env.example .env
# Ajouter vos clés API dans .env

python web_app.py
