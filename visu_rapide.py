#!/usr/bin/env python3
"""
Visualisation rapide avec mouvement visible
"""

import numpy as np
from Corps_numba import GalaxieNumba, G
from visualizer3d_sans_vbo import Visualizer3D

# Constante pour accélérer le mouvement
FACTEUR_VITESSE = 10000

# Création de la galaxie
n_etoiles = 100
n_corps = n_etoiles + 1
galaxie = GalaxieNumba(n_corps)

# Trou noir
galaxie.ajouter_trou_noir(1e6, np.zeros(3))

print("Création des étoiles...")
for i in range(1, n_corps):
    r = np.random.uniform(2, 5)
    theta = np.random.uniform(0, 2*np.pi)
    
    pos = np.array([
        r * np.cos(theta),
        r * np.sin(theta),
        np.random.uniform(-0.3, 0.3)
    ])
    
    # Vitesse orbitale multipliée par FACTEUR_VITESSE
    v = np.sqrt(G * 1e6 / r) * FACTEUR_VITESSE
    vitesse = np.array([
        -v * np.sin(theta),
        v * np.cos(theta),
        0
    ])
    
    masse = np.random.uniform(0.5, 10)
    galaxie.ajouter_etoile(i, masse, pos, vitesse)

# Visualisation
visualizer = Visualizer3D(
    points=galaxie.get_points(),
    colors=galaxie.get_couleurs(),
    luminosities=galaxie.get_luminosites(),
    bounds=((-6, 6), (-6, 6), (-1, 1))
)

visualizer.camera_distance = 12

def updater(dt):
    # Multiplie dt pour accélérer encore
    galaxie.update(dt * 50, mode='numba_parallel')
    return galaxie.get_points(), galaxie.get_couleurs(), galaxie.get_luminosites()

print("\n" + "="*50)
print("🎮 VISUALISATION RAPIDE")
print("="*50)
print("✅ Les étoiles devraient maintenant TOURNER visiblement !")
print("🖱️  Clic gauche + souris : rotation")
print("🖱️  Molette : zoom")
print("❌ ECHAP : quitter")
print("="*50)

visualizer.run_with_updater(updater, dt=0.02)