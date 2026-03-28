#!/usr/bin/env python3
"""
Test simple pour vérifier que les étoiles bougent
"""

import numpy as np
from Corps_numba import GalaxieNumba, G

# Création d'une petite galaxie
n_etoiles = 5
n_corps = n_etoiles + 1
galaxie = GalaxieNumba(n_corps)

# Trou noir
galaxie.ajouter_trou_noir(1e6, np.zeros(3))

# Étoiles
print("Positions initiales:")
for i in range(1, n_corps):
    r = 3.0
    theta = i * 2*np.pi / n_etoiles  # Réparties en cercle
    pos = np.array([r*np.cos(theta), r*np.sin(theta), 0])
    
    # Vitesse pour orbite circulaire
    v = np.sqrt(G * 1e6 / r)
    vitesse = np.array([-v*np.sin(theta), v*np.cos(theta), 0])
    
    masse = 1.0
    galaxie.ajouter_etoile(i, masse, pos, vitesse)
    print(f"Étoile {i}: position={pos}, vitesse={vitesse}")

# Simulation de quelques pas
print("\nSimulation de 10 pas de temps:")
for step in range(10):
    galaxie.update(0.1, mode='numba_parallel')
    print(f"Pas {step}:")
    for i in range(1, n_corps):
        print(f"  Étoile {i}: {galaxie.positions[i]}")