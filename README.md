# Implementation-et-simulation-de-modeles-de-decision-chez-les-pnh

# Simulation d'un Algorithme Génétique appliqué à la Prospect Theory

Ce projet a été réalisé dans le cadre de mon stage. Il propose une simulation numérique en Python (développée sous Spyder) combinant un **algorithme génétique** et les concepts de la **Théorie des Perspectives (Prospect Theory)**.

## Description du projet

L'objectif est de faire évoluer une population d'agents virtuels confrontés à des choix économiques (des loteries) pour observer comment leurs comportements et leurs préférences face au risque s'adaptent au fil des générations. La simulation sépare l'étude en deux contextes distincts : **le mécanisme de gains** et **le mécanisme de pertes**.

Chaque agent est caractérisé par trois gènes (paramètres comportementaux) :
* **Alpha ($\alpha$)** : Distorsion des probabilités (fonction de pondération de Prelec).
* **Beta ($\beta$)** : Attitude face au risque (courbure de la fonction d'utilité).
* **k** : Sensibilité au différentiel d'utilité (paramètre de la sigmoïde de choix stochastique).

##  Fonctionnalités du code

* **Initialisation** d'une population aléatoire de 100 agents.
* **Boucle évolutive complète** sur 500 générations (Évaluation, Sélection des élites, Reproduction par interpolation lineaire, Mutation).
* **Visualisation graphique complète** via `matplotlib` :
    * Heatmaps de l'évolution des gènes au fil du temps.
    * Graphiques des fonctions d'utilité et de pondération (populations initiales vs finales).
    * Analyse de l'impact du paramètre $k$ sur les choix des agents.

##  Prérequis

Pour exécuter ce script, vous aurez besoin de Python et des bibliothèques suivantes :
* `numpy`
* `matplotlib`
