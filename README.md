# Projet de Modélisation et Clustering de Données Textuelles

Ce repository contient le code source pour développer un modèle de clustering basé sur la réduction de la dimensionalité à l'aide de l'ACP, UMAP et t-SNE, combiné avec l'algorithme de clustering k-means.

## Objectif

L'objectif est de prendre des données textuelles (en utilisant les données NG20 comme présenté dans le template, avec seulement 2000 documents) et d'appliquer une approche tandem ou séquentielle en utilisant l'ACP, UMAP et t-SNE, suivi du clustering avec l'algorithme k-means.

## Structure du Repository

- `main.py`: Fichier principal contenant le code pour évaluer chaque approche (ACP+kmeans, UMAP+kmeans, t-SNE+kmeans) en utilisant les métriques NMI et ARI  à partir des classes connues.
- `requirements.txt`: Les librairies pré-requis.
- `experiments/`: Dossier contenant les notebooks de l'implémentation de chaque méthode de réduction de dimensions ave le clsuterinng en utilisant l'algorithme k-means.
- `DockerFile`: les lignes nécessaires pour créer une image Docker.
- `embeddings.csv`: Fichier csv contenant les données textuelles nécessaires au projet.

## Utilisation locale

1. Clonez le repository:

   ```bash
   git clone https://github.com/haf2000/Exam_Data_Eng.git
   cd Exam_Data_Eng.

2. cd  /Exam_Data_Eng

3. Executez la commande : docker build -f DockerFile -t main_image:v1 .

4. Executez : docker run --name conteneur_main main_image:v1

## Utilisation avec DockerHub

1. git pull l'image de DockerHub en exécutant : docker pull (lien communiqué)

2. docker run --name conteneur_main main_image:v1

## Visualisation des données sur un plan à l'aide de T-sne, UMAP et ACP

La visualisation est dans les notebooks dans Experiments
