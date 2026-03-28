# Projet_Python_NCorps
Projet Python NCorps

## 1. Présentation du Projet
Ce projet, réalisé dans le cadre du Master 2 (2026), a pour objectif de simuler la dynamique d'une galaxie en résolvant le problème classique à $N$-corps mûs par la gravité. La simulation prend en compte $N$ étoiles gravitant autour d'un trou noir supermassif central.

Le défi technique consiste à optimiser le calcul des interactions gravitationnelles ($O(N^2)$) pour permettre la simulation de milliers d'étoiles de manière fluide.

## 2. Modèle Physique et Données

### Lois du mouvement
L'accélération $\vec{a}_i$ de chaque étoile $i$ est calculée selon la loi de gravitation universelle de Newton :

$$
\left\lbrace
\begin{array}{lcl}
\vec{f}_{i}(t) & = & m_{i}.\vec{a}_{i}(t) \\
\vec{a}_{i}(t) & = & \displaystyle \sum_{j\neq i} \mathcal{G}\frac{m_{j}}{\|\vec{p}_{j}(t)-\vec{p}_{i}(t)\|^{3}}(\vec{p}_{j}(t)-\vec{p}_{i}(t))
\end{array}
\right.
$$

Où $\mathcal{G} = 1.560339 \cdot 10^{-13}$. Les unités sont :
- **Distances** : Années-lumière (ly)
- **Masses** : Masses solaires ($M_{\odot}$)
- **Temps** : Années terrestres (yr)

L'intégration numérique (mise à jour de la position et vitesse) suit le schéma suivant :
- $\vec{v}_{i}(t+\delta_{t}) = \vec{v}_{i}(t) + \delta t \vec{a}_{i}(t)$
- $\vec{p}_{i}(t+\delta_{t}) = \vec{p}_{i}(t)+\delta t \vec{v}_{i}(t) + \frac{1}{2}(\delta t)^{2}\vec{a}_{i}(t)$

### Classification Stellaire
La couleur des étoiles est déterminée par leur masse $m_i$ :
- **$m_i > 5$** : Géante bleue (150, 180, 255)
- **$m_i > 2$** : Étoile blanche (255, 255, 255)
- **$m_i \geq 1$** : Étoile type Soleil (255, 255, 200)
- **Sinon** : Naine rouge (250, 150, 100)
- **Trou Noir** : Noir (0, 0, 0)

## 3. Approches et Optimisations

Le projet explore quatre niveaux d'optimisation pour augmenter le nombre de corps simulés :

1.  **Version Naïve Objet (`Corps_accel.py`)** : Utilisation de classes Python. Approche pédagogique mais limitée par la lenteur de l'interpréteur Python sur les boucles imbriquées ($O(N^2)$).
2.  **Version Vectorisée (`Corps_vectorise.py`)** : Utilisation intensive de **NumPy** et du *broadcasting* pour éliminer les boucles Python. Très rapide pour $N < 2000$, mais limitée par l'occupation mémoire en $O(N^2)$.
3.  **Version JIT Numba (`Corps_numba.py`)** : Compilation Just-In-Time du code Python en code machine. Utilisation de la parallélisation (`prange`) sur tous les cœurs du processeur.
4.  **Version Barnes-Hut (`Corps_BarnesHut.py`)** : Implémentation d'un **Octree** (Arbre spatial). Cette méthode réduit la complexité de $O(N^2)$ à **$O(N \log N)$** en approximant les groupes d'étoiles lointaines par leur centre de masse.

## 4. Visualisation 3D
La visualisation utilise **SDL2** et **OpenGL** pour un rendu fluide :
- **Rotation** : Clic gauche + déplacement souris.
- **Zoom** : Molette de la souris.
- **Navigation** : Rendu dynamique des points avec luminosité proportionnelle à la masse.

## 5. Benchmarks de Performance

Les mesures ont été effectuées avec le script `test_perf.py` (moyenne sur 10 itérations) :

| Nombre d'étoiles (N) | Naïve (Objet) | Vectorisée (NumPy) | Numba (Parallel) | Barnes-Hut |
| :--- | :--- | :--- | :--- | :--- |
| **100** | [VOS_SEC] s | [VOS_SEC] s | [VOS_SEC] s | [VOS_SEC] s |
| **1 000** | [VOS_SEC] s | [VOS_SEC] s | [VOS_SEC] s | [VOS_SEC] s |
| **10 000** | N/A | [VOS_SEC] s | [VOS_SEC] s | [VOS_SEC] s |
| **100 000** | N/A | N/A | N/A | [VOS_SEC] s |

## 6. Installation et Utilisation

### Prérequis
- Python 3.10+
- `numpy`, `numba`, `pysdl2`, `pysdl2-dll`, `PyOpenGL`

### Commandes
- **Lancer le benchmark** :
  ```bash
  python test_perf.py
  ```
- **Lancer la simulation Barnes-Hut** (Haute performance) :
  ```bash
  python Corps_BarnesHut.py
  ```
- **Lancer la simulation Numba** (Parallélisée) :
  ```bash
  python Corps_numba.py --visualize
  ```

## 7. Analyse et Conclusion
L'optimisation via **Numba** permet un gain de performance massif par rapport à la version naïve (jusqu'à 100x plus rapide). Cependant, pour des galaxies dépassant les 10 000 étoiles, seul l'algorithme de **Barnes-Hut** permet de maintenir un taux de rafraîchissement (FPS) acceptable grâce à sa réduction de complexité algorithmique.

***