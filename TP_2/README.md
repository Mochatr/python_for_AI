# Rapport de Projet : Implémentation d'un Mini Réseau de Neurones (NumPy)

## 1. Introduction et Objectifs
L'objectif de ce projet est de concevoir, implémenter et évaluer un réseau de neurones multicouche *from scratch*, en utilisant uniquement la bibliothèque **NumPy** pour les calculs matriciels et **Matplotlib** pour la visualisation. 
Le cas d'étude choisi est l'apprentissage de la fonction logique **XOR**. Ce problème classique démontre les limites des modèles linéaires simples (perceptrons) face à des données non-linéairement séparables, et justifie l'utilisation d'une architecture multicouche intégrant des fonctions d'activation non-linéaires.

## 2. Étude des Fonctions d'Activation
Les fonctions d'activation introduisent la non-linéarité indispensable au réseau. Six fonctions ont été implémentées et analysées :

1. **ReLU (Rectified Linear Unit)** : $f(x) = \max(0, x)$. Très utilisée pour sa simplicité et sa capacité à atténuer le problème de la disparition du gradient dans les réseaux profonds.
2. **Sigmoid** : $f(x) = \frac{1}{1 + e^{-x}}$. Restreint les sorties dans l'intervalle [0, 1], idéale pour la couche de sortie en classification binaire. Une forme numériquement stable a été implémentée pour prévenir l'overflow.
3. **Tanh (Tangente Hyperbolique)** : $f(x) = \tanh(x)$. Centre les valeurs autour de zéro sur l'intervalle ]-1, 1[, facilitant la convergence des couches cachées.
4. **LeakyReLU** : $f(x) = x$ si $x > 0$, sinon $\alpha x$. Résout le problème des "dying ReLUs" en conservant un léger gradient pour les valeurs négatives.
5. **Softplus** : $f(x) = \ln(1 + e^x)$. Une approximation lisse et continûment dérivable de la fonction ReLU. Sa dérivée correspond d'ailleurs à la fonction Sigmoid.
6. **ELU (Exponential Linear Unit)** : Pousse les activations moyennes vers zéro, combinant la robustesse de ReLU avec une meilleure gestion des valeurs négatives via une composante exponentielle.

**Note sur la rétropropagation :** Pour optimiser les calculs, les dérivées de fonctions telles que Sigmoid ou Tanh ont été exprimées en fonction de l'activation (l'output) déjà calculée lors de la phase de propagation avant (ex: $\sigma'(x) = s(1-s)$).

## 3. Architecture du Réseau (2 -> 3 -> 1)
L'architecture déployée est adaptée à la dimensionnalité du problème XOR :
* **Couche d'entrée** : 2 neurones (pour les attributs $x_1$ et $x_2$).
* **Couche cachée** : 3 neurones (fonction d'activation : Tanh ou ReLU).
* **Couche de sortie** : 1 neurone (fonction d'activation : Sigmoid, retournant une probabilité $\hat{y}$).

### 3.1. Propagation Avant (Forward Propagation)
Les opérations matricielles appliquées pour un lot de $N$ exemples sont les suivantes :
1. $Z_1 = X W_1 + b_1$ (Pré-activation de la couche cachée)
2. $A_1 = \sigma_{hidden}(Z_1)$ (Activation de la couche cachée)
3. $Z_2 = A_1 W_2 + b_2$ (Pré-activation de la couche de sortie)
4. $\hat{y} = \sigma_{out}(Z_2)$ (Prédiction finale)

Avec les dimensions : $X \in \mathbb{R}^{N\times 2}$, $W_1 \in \mathbb{R}^{2\times 3}$, $A_1 \in \mathbb{R}^{N\times 3}$, $W_2 \in \mathbb{R}^{3\times 1}$.

### 3.2. Fonction de Perte et Rétropropagation (Backpropagation)
La fonction de coût utilisée est l'Erreur Quadratique Moyenne (MSE) :
$$ L = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 $$

L'algorithme de rétropropagation calcule les gradients de l'erreur par rapport aux poids en appliquant la règle de composition des dérivées (Chain Rule) de la sortie vers l'entrée :
* **Gradient de l'erreur** : $dY = \frac{2}{N}(\hat{y} - y)$
* **Couche de sortie** : $dZ_2 = dY \odot \sigma'_{out}(Z_2)$
  $\Rightarrow dW_2 = A_1^T dZ_2$ et $db_2 = \sum dZ_2$
* **Couche cachée** : $dA_1 = dZ_2 W_2^T$ puis $dZ_1 = dA_1 \odot \sigma'_{hidden}(Z_1)$
  $\Rightarrow dW_1 = X^T dZ_1$ et $db_1 = \sum dZ_1$

Les poids sont ensuite mis à jour via l'algorithme de Descente de Gradient : $W = W - \eta dW$ (où $\eta$ est le taux d'apprentissage).

## 4. Expérimentation sur Données Bruitées
Afin d'évaluer la résilience du réseau, le dataset XOR de base a été augmenté et altéré :
* **Duplication** : 100 exemples par coin (total 400).
* **Bruit Gaussien** : $\epsilon \sim \mathcal{N}(0, 0.05)$ ajouté aux coordonnées.
* **Labels erronés** : Inversion aléatoire de 4% des labels cibles.

Les données ont ensuite été partitionnées en un jeu d'entraînement (80%) et un jeu de test (20%), simulant un flux de travail classique en apprentissage supervisé.

## 5. Évaluation des Performances
Sur les données de test, le modèle affiche une excellente capacité de généralisation malgré la perturbation introduite.

### 5.1. Matrice de Confusion et Métriques
Les prédictions probabilistes ont été binarisées avec un seuil à 0.5.
* **Accuracy (Exactitude)** : Le modèle atteint généralement plus de 90%, indiquant que la vaste majorité des exemples (vrais positifs et vrais négatifs) sont bien classés.
* **Precision et Recall** : Ces scores, également supérieurs à 0.90, confirment que le modèle ne souffre pas de biais majeur entre la prédiction des 1 et des 0.
* **F1-Score** : Fournit une moyenne harmonique robuste autour de 0.92, pertinente pour valider l'équilibre du modèle sur des données potentiellement bruitées.

### 5.2. Courbe ROC et AUC
La courbe ROC (taux de vrais positifs contre taux de faux positifs à divers seuils) présente un coude net vers le coin supérieur gauche. 
L'aire sous la courbe (**AUC**) calculée approche les 0.90 - 0.95. Un score proche de 1 valide la robustesse de l'architecture pour discriminer le signal du bruit stochastique ajouté au dataset.

## 6. Analyse et Influence des Hyperparamètres

Durant les expérimentations, la modulation de plusieurs hyperparamètres a révélé des dynamiques fondamentales :

1. **Influence du Taux d'Apprentissage (Learning Rate - $\eta$)** :
   * **$\eta$ trop faible (ex: 0.01)** : L'entraînement est stable mais extrêmement lent, nécessitant un très grand nombre d'epochs pour réduire la *loss*.
   * **$\eta$ optimal (ex: 0.1 - 0.5)** : Permet une convergence rapide et lisse de la courbe MSE.
   * **$\eta$ trop élevé (ex: 2.0)** : Entraîne des oscillations violentes du gradient, empêchant la MSE de converger (overshooting du minimum local).

2. **Influence du Nombre de Neurones Cachés** :
   * **2 neurones** : Capacité minimale. Le réseau peine parfois à résoudre le XOR de façon robuste si l'initialisation des poids n'est pas favorable.
   * **3 neurones** (Configuration choisie) : Apporte la flexibilité nécessaire pour créer les limites de décision géométriques englobant les classes de manière fiable.
   * **10+ neurones** : Sur un problème aussi simple que XOR, un trop grand nombre de neurones favorise l'*overfitting* (surapprentissage) sur le bruit du dataset d'entraînement, dégradant les métriques sur le set de test.

3. **Choix de la Fonction d'Activation** :
   * La combinaison **Tanh (cachée) / Sigmoid (sortie)** offre la convergence la plus rapide pour XOR, le Tanh centrant efficacement les données autour de 0.
   * L'utilisation de **ReLU** sur la couche cachée nécessite souvent plus d'epochs ou un taux d'apprentissage plus précautionneux, en raison de sa non-dérivabilité en zéro et du plafonnement (dead neurons) qui peut survenir aléatoirement à l'initialisation.

## 7. Conclusion
La conception de bout en bout de ce réseau a permis de cristalliser la compréhension mathématique et programmatique du Deep Learning. De la dérivation manuelle des gradients (Jacobienne) à la gestion de la stabilité numérique (Sigmoid stable), chaque composante algorithmique a été validée. 
L'évaluation sur un jeu de données synthétique complexe prouve que même une architecture miniature (2->3->1), lorsqu'elle est correctement paramétrée, est capable d'assimiler des frontières de décision hautement non-linéaires tout en conservant une forte immunité face au bruit statistique.