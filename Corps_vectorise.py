#!/usr/bin/env python3
"""
VERSION 2 - VECTORISÉE SANS CLASSE
Problème à N corps avec calculs vectorisés (NumPy)
Utilise uniquement des tableaux et des fonctions
"""

import numpy as np
import time

# Constante gravitationnelle
G = 1.560339e-13

# ============================================================
# FONCTIONS DE CRÉATION ET GESTION DES DONNÉES
# ============================================================

def creer_galaxie(n_corps):
    """
    Crée les tableaux pour une galaxie de n_corps corps
    
    Args:
        n_corps (int): Nombre total de corps
    
    Returns:
        tuple: (positions, vitesses, masses, couleurs)
    """
    positions = np.zeros((n_corps, 3), dtype=np.float64)
    vitesses = np.zeros((n_corps, 3), dtype=np.float64)
    masses = np.zeros(n_corps, dtype=np.float64)
    couleurs = np.zeros((n_corps, 3), dtype=np.float32)  # RGB 0-255
    
    return positions, vitesses, masses, couleurs


def ajouter_trou_noir(positions, vitesses, masses, couleurs, masse, position, vitesse=np.zeros(3)):
    """
    Ajoute le trou noir à l'indice 0
    """
    masses[0] = masse
    positions[0] = position
    vitesses[0] = vitesse
    couleurs[0] = [0, 0, 0]  # Noir
    return positions, vitesses, masses, couleurs


def ajouter_etoile(positions, vitesses, masses, couleurs, idx, masse, position, vitesse=np.zeros(3)):
    """
    Ajoute une étoile à l'indice spécifié
    """
    masses[idx] = masse
    positions[idx] = position
    vitesses[idx] = vitesse
    couleurs[idx] = _couleur_selon_masse(masse)
    return positions, vitesses, masses, couleurs


def _couleur_selon_masse(masse):
    """Détermine la couleur selon la masse"""
    if masse > 5:
        return np.array([150, 180, 255], dtype=np.float32)  # Bleu-blanc
    elif masse > 2:
        return np.array([255, 255, 255], dtype=np.float32)  # Blanc
    elif masse > 1:
        return np.array([255, 255, 200], dtype=np.float32)  # Jaune
    else:
        return np.array([250, 150, 100], dtype=np.float32)  # Rouge


# ============================================================
# FONCTIONS DE CALCUL VECTORISÉ
# ============================================================

def calculer_accelerations_vectorisees(positions, masses):
    """
    Version VECTORISÉE du calcul des accélérations
    (Utilise le broadcasting NumPy - PAS DE BOUCLES !)
    
    Args:
        positions (np.ndarray): Tableau des positions (N, 3)
        masses (np.ndarray): Tableau des masses (N,)
    
    Returns:
        np.ndarray: Accélérations (N, 3)
    """
    n = len(masses)
    
    # Créer des matrices 3D pour le broadcasting
    # positions_i : (n, 1, 3) - chaque corps i
    # positions_j : (1, n, 3) - tous les corps j
    pos_i = positions[:, np.newaxis, :]  # Shape: (n, 1, 3)
    pos_j = positions[np.newaxis, :, :]  # Shape: (1, n, 3)
    
    # Vecteurs de i vers j : (n, n, 3)
    r_vectors = pos_j - pos_i
    
    # Distances entre tous les couples (i,j) : (n, n)
    # +1e-10 pour éviter la division par zéro
    distances = np.linalg.norm(r_vectors, axis=2) + 1e-10
    
    # m_j / distance^3 pour tous les couples : (n, n)
    m_sur_r3 = masses[np.newaxis, :] / (distances**3)
    
    # Ignorer l'auto-interaction (quand i == j)
    np.fill_diagonal(m_sur_r3, 0)
    
    # Calcul vectorisé des accélérations :
    # a_i = G * somme( m_j * (r_j - r_i) / |r_j - r_i|^3 )
    accelerations = G * np.sum(
        m_sur_r3[:, :, np.newaxis] * r_vectors,
        axis=1
    )  # Shape finale: (n, 3)
    
    return accelerations


def mise_a_jour(positions, vitesses, masses, dt):
    """
    Met à jour positions et vitesses de TOUS les corps en une seule opération
    
    Args:
        positions (np.ndarray): Positions actuelles (N, 3)
        vitesses (np.ndarray): Vitesses actuelles (N, 3)
        masses (np.ndarray): Masses (N,)
        dt (float): Pas de temps
    
    Returns:
        tuple: (nouvelles_positions, nouvelles_vitesses)
    """
    # Calcul vectorisé des accélérations
    accelerations = calculer_accelerations_vectorisees(positions, masses)
    
    # Mise à jour vectorisée (pas de boucle!)
    nouvelles_vitesses = vitesses + accelerations * dt
    nouvelles_positions = positions + vitesses * dt + 0.5 * accelerations * dt**2
    
    return nouvelles_positions, nouvelles_vitesses


# ============================================================
# FONCTIONS DE GESTION DES DONNÉES POUR VISUALISATION
# ============================================================

def get_points(positions):
    """Retourne les positions pour visualisation (float32)"""
    return positions.astype(np.float32)


def get_couleurs(couleurs):
    """Retourne les couleurs pour visualisation"""
    return couleurs.astype(np.float32)


def get_masses(masses):
    """Retourne les masses"""
    return masses.astype(np.float32)


def get_luminosites(masses):
    """Retourne les luminosités basées sur la masse"""
    max_masse = masses.max()
    if max_masse > 0:
        return np.clip(masses / max_masse, 0.3, 1.0).astype(np.float32)
    return np.ones(len(masses), dtype=np.float32) * 0.5


# ============================================================
# FONCTION PRINCIPALE DE VISUALISATION
# ============================================================

def visualisation_vectorisee():
    """
    Fonction principale pour la version vectorisée avec visualisation
    """
    import os
    
    # Paramètres
    n_etoiles = 100
    n_corps = n_etoiles + 1
    
    print("\n" + "="*60)
    print("VERSION 2 - VECTORISÉE (SANS CLASSE)")
    print("="*60)
    
    # Création des tableaux
    positions, vitesses, masses, couleurs = creer_galaxie(n_corps)
    
    # Ajout du trou noir
    positions, vitesses, masses, couleurs = ajouter_trou_noir(
        positions, vitesses, masses, couleurs,
        masse=1e6,
        position=np.zeros(3)
    )
    
    # Ajout des étoiles
    print("Création des étoiles...")
    for i in range(1, n_corps):
        r = np.random.uniform(2, 5)  # Rayon entre 2 et 5
        theta = np.random.uniform(0, 2*np.pi)
        
        pos = np.array([
            r * np.cos(theta),
            r * np.sin(theta),
            np.random.uniform(-0.3, 0.3)  # Épaisseur du disque
        ])
        
        # Vitesse orbitale RÉALISTE (sans facteur)
        v = np.sqrt(G * 1e6 / r)  # Vitesse réelle en ly/an
        
        # On garde la vitesse réelle, on ajustera le dt dans l'updater
        vitesse = np.array([
            -v * np.sin(theta),
            v * np.cos(theta),
            0
        ])
        
        masse = np.random.uniform(0.5, 10)
        positions, vitesses, masses, couleurs = ajouter_etoile(
            positions, vitesses, masses, couleurs,
            i, masse, pos, vitesse
        )
    
    # Vérification des vitesses
    print(f"Vitesse typique: {np.linalg.norm(vitesses[1]):.2e} ly/an")
    
    # Préparation pour la visualisation
    points_init = get_points(positions)
    lum = get_luminosites(masses)
    
    # Calcul des limites (ajustées)
    x_min, x_max = -8, 8
    y_min, y_max = -8, 8
    z_min, z_max = -2, 2
    bounds = ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    
    print(f"Nombre de corps: {n_corps}")
    print(f"Limites: {bounds}")
    print(f"Positions initiales - min: {points_init.min(axis=0)}, max: {points_init.max(axis=0)}")
    
    # Visualisation
    try:
        from visualizer3d_sans_vbo import Visualizer3D
        
        visualizer = Visualizer3D(
            points=points_init,
            colors=couleurs,
            luminosities=lum,
            bounds=bounds
        )
        
        visualizer.camera_distance = 15
        
        # Compteur pour le débogage
        frame_count = 0
        
        def updater(dt):
            nonlocal positions, vitesses, frame_count
            frame_count += 1
            
            # AUGMENTEZ CE FACTEUR POUR VOIR LE MOUVEMENT !
            facteur_temps = 500  # Au lieu de 10
            
            positions, vitesses = mise_a_jour(positions, vitesses, masses, dt * facteur_temps)
            
            if frame_count % 30 == 0:
                pos_max = np.max(np.abs(positions))
                print(f"Max distance: {pos_max:.2f} | "
                    f"Étoile 1: ({positions[1,0]:.2f}, {positions[1,1]:.2f})")
            
            return (get_points(positions),
                get_couleurs(couleurs),
                get_luminosites(masses))
        
        print("\n🎮 Visualisation avec version vectorisée (sans classe)")
        print(f"   Facteur de temps: 10")
        print("   Les étoiles devraient tourner sans s'échapper !")
        
        visualizer.run_with_updater(updater, dt=0.01)
        
    except Exception as e:
        print(f"Erreur de visualisation: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# FONCTION DE BENCHMARK
# ============================================================

def benchmark_vectorise(n_etoiles=100, n_iterations=50, dt=0.01):
    """
    Benchmark de la version vectorisée
    """
    n_corps = n_etoiles + 1
    positions, vitesses, masses, couleurs = creer_galaxie(n_corps)
    
    # Initialisation
    positions, vitesses, masses, couleurs = ajouter_trou_noir(
        positions, vitesses, masses, couleurs, 1e6, np.zeros(3)
    )
    
    for i in range(1, n_corps):
        r = np.random.uniform(1, 5)
        theta = np.random.uniform(0, 2*np.pi)
        pos = np.array([r*np.cos(theta), r*np.sin(theta), 0])
        masse = np.random.uniform(0.5, 10)
        positions, vitesses, masses, couleurs = ajouter_etoile(
            positions, vitesses, masses, couleurs, i, masse, pos
        )
    
    # Mesure du temps
    start = time.time()
    for _ in range(n_iterations):
        positions, vitesses = mise_a_jour(positions, vitesses, masses, dt)
    end = time.time()
    
    return end - start


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        t = benchmark_vectorise(n_etoiles=n, n_iterations=30)
        print(f"Temps pour {n} étoiles: {t:.4f} secondes")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--visualize":
        visualisation_vectorisee()
    
    else:
        print("Usage:")
        print("  python3 Corps_vectorise.py --benchmark [n_etoiles]")
        print("  python3 Corps_vectorise.py --visualize")