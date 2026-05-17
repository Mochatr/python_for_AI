# TP - Visualisation des Fonctions d'Activation (NumPy + Matplotlib)

## Description du Projet
Ce projet consiste à implémenter et à visualiser graphiquement plusieurs fonctions d'activation couramment utilisées dans le domaine du deep learning. L'objectif principal est d'analyser leurs formes et leurs propriétés à l'aide de calculs entièrement vectorisés avec NumPy et Matplotlib.

## Objectifs Pédagogiques
À la fin de ce travail, les notions suivantes sont maîtrisées :
* Création d'un vecteur d'échantillonnage régulier avec `np.linspace`.
* Codage de fonctions mathématiques de manière vectorisée, garantissant l'absence de boucles Python sur les points de données.
* Organisation et parcours des structures de données (fonctions, couleurs, marqueurs) via des dictionnaires synchronisés.
* Création de visualisations claires et professionnelles intégrant des titres, axes, grilles et légendes configurés avec soin.
* Utilisation avancée des grilles de sous-figures (`subplots`) personnalisées (couleurs, types de marqueurs, échantillonnage des repères visuels avec `markevery`).

---

## Pré-requis
L'environnement d'exécution requiert les dépendances suivantes :
* **Python 3.x**
* **NumPy**
* **Matplotlib**

---

## Fonctions d'Activation Implémentées

Les fonctions de la partie principale ainsi que les extensions mathématiques ont été intégrées de façon vectorisée :

| Fonction | Équation Mathématique | Particularité de l'implémentation |
| :--- | :--- | :--- |
| **ReLU** | $$f(x) = \max(0, x)$$ | Utilisation de `np.maximum` |
| **Sigmoid** | $$f(x) = \frac{1}{1 + e^{-x}}$$ | Utilisation de `np.exp` |
| **Tanh** | $$f(x) = \tanh(x)$$ | Utilisation de `np.tanh` |
| **LeakyReLU** | $$f(x) = \begin{cases} x & \text{si } x > 0 \\ a \cdot x & \text{sinon} \end{cases}$$ avec $a = 0.1$ | Utilisation de `np.where` |
| **Softplus** (Extension) | $$f(x) = \ln(1 + e^x)$$ | Utilisation de `np.log` et `np.exp` |
| **ELU** (Extension) | $$f(x) = \begin{cases} x & \text{si } x > 0 \\ \alpha(e^x - 1) & \text{sinon} \end{cases}$$ avec $\alpha = 1$ | Utilisation de `np.where` et `np.exp` |

---

## Structure du Code et Visualisations

Le projet est articulé autour de deux types majeurs de représentations visuelles :
1. **Tracé comparatif global** : Superposition de l'ensemble des courbes sur un graphique unique afin de comparer directement leurs amplitudes et leurs zones de saturation.
2. **Grille de Subplots** : Isolation de chaque fonction sur son propre repère orthogonal au sein d'une grille propre, facilitant l'observation individuelle sans chevauchement.

Chaque courbe exploite un style graphique unique (couleurs de la palette *Tableau* et marqueurs géométriques différenciés) partagé de manière cohérente à travers des clés de dictionnaires uniques. Un pas d'affichage des marqueurs (`markevery=60`) a été appliqué pour conserver une parfaite lisibilité sur les 600 points du vecteur $x$.

---

## Livrables Générés
Le dossier projet contient les éléments obligatoires suivants :
* **`TP_Activation.ipynb`** : Le Jupyter Notebook contenant le code source.
* **`activations_all.png`** : Graphique comparatif des 4 fonctions initiales.
* **`activations_subplots.png`** : Grille 2x2 présentant individuellement les 4 fonctions initiales.
* **`activations_all_extension.png`** : Graphique comparatif incluant les 6 fonctions (avec Softplus et ELU).
* **`activations_subplots_extension.png`** : Grille contenant les 6 fonctions d'activation avec leurs extensions respectives.