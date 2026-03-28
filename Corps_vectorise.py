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
    grid_pos = np.zeros((n_corps, 2), dtype=np.int8)
    gravity_centers = np.zeros((20,20,3), dtype= np.float16) 
    
    return positions, vitesses, masses, couleurs, grid_pos, gravity_centers

def belongs_to(position):
    # CORRECTION : clamp pour éviter NameError si la position sort du domaine [-10, 10]
    x_coor = np.clip(position[0], -10, 10)
    y_coor = np.clip(position[1], -10, 10)
    #print("x_coor :", x_coor)
    #print("y_coor :", y_coor)

    # CORRECTION : initialisation de ix et iy pour éviter NameError
    ix, iy = 0, 0

    for i in range(0,20) :
        inf = -10 + i*( 10 + 10)/19
        sup = -10 + (i+1)*( 10 + 10)/19
        if x_coor >= inf and x_coor <= sup :
            ix = i
        if y_coor >= inf and y_coor <= sup :
            iy = i
    
    grid_index = np.zeros((2))
    grid_index[0] = ix 
    grid_index[1] = iy
    return grid_index

# CORRECTION : belongs_vect supprimée car inutilisable telle qu'écrite
# np.apply_along_axis(belongs_to, 1, positions) est utilisé à la place


def ajouter_trou_noir(positions, vitesses, masses, grid_pos, couleurs, masse, position, vitesse=np.zeros(3)):
    """
    Ajoute le trou noir à l'indice 0
    """
    masses[0] = masse
    positions[0] = position
    vitesses[0] = vitesse
    couleurs[0] = [0, 0, 0]  # Noir
    grid_pos[0] = belongs_to(position)
    return positions, vitesses, masses, couleurs, grid_pos


def ajouter_etoile(positions, vitesses, masses, grid_pos, couleurs, idx, masse, position, vitesse=np.zeros(3)):
    """
    Ajoute une étoile à l'indice spécifié
    """
    masses[idx] = masse
    positions[idx] = position
    vitesses[idx] = vitesse
    couleurs[idx] = _couleur_selon_masse(masse)
    grid_pos[idx] = belongs_to(position)
    return positions, vitesses, masses, couleurs, grid_pos


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
    
def update_gravity_centers(n_corps, positions,grid_pos,gravity_centers):
        for i in range(20):
            for j in range(20):
                gravity_centers[i,j] = compute_gravity_center(n_corps,positions,grid_pos, i,j)
        return gravity_centers


def compute_gravity_center(n_corps,positions,grid_pos,coord_x,coord_y): 
    #print("coord_x :", coord_x)
    #print("coord_y :", coord_y)
    loc_pos = []
    loc_masses = []  # CORRECTION : stocker les masses pour le centre de gravité pondéré
    #print("grid_pos :", grid_pos.shape)
    for i in range(n_corps):
        #print("self.grid[i,0] :", grid_pos[i,0])
        #print("self.grid[i,1] :", grid_pos[i,1])
        #print("self.grid[i,0] == coord_x :", self.grid[i,0] == coord_x )
        #print("self.grid[i,1] == coord_y :", self.grid[i,1] == coord_y )

        if int(grid_pos[i,0]) == coord_x and int(grid_pos[i,1]) == coord_y :
            loc_pos.append(positions[i])
            loc_masses.append(masses[i])  # CORRECTION : collecter les masses
    loc_pos = np.array(loc_pos)
    loc_masses = np.array(loc_masses)
    #print("loc_pos :",loc_pos)
    #print("np.mean(loc_pos) :", np.mean(loc_pos,axis =0))
    # CORRECTION : moyenne pondérée par les masses (centre de gravité réel)
    return np.average(loc_pos, axis=0, weights=loc_masses) if np.shape(loc_pos)[0] > 0 else np.zeros((3,))


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

def calculer_accelerations_vectorisees_2(n_corps, positions, masses, grid_pos, gravity_centers):
        "Grid"
        accelerations = np.zeros((n_corps, 3), dtype=np.float64)
        #self.gravity_centers = np.ones((20,20,3), dtype=np.int16)
        gravity_centers = update_gravity_centers(n_corps, positions, grid_pos, gravity_centers)
        #print("self.grid :", self.grid)
        #print("self.gravity_centers :", self.gravity_centers[6:11, 6:12,:])
        for i in range(n_corps):
            #print(" star id :", i)
            for j in range(n_corps):
                if i != j:
                    #print("self.grid[j,0] :", self.grid[j,0])
                    #print("self.grid[j,1] :", self.grid[j,1])
                    #print("j", j)
                    center_j = gravity_centers[int(grid_pos[j,0]), int(grid_pos[j,1])]
                    if 1/2* np.linalg.norm(positions[i] - center_j) > 0.3 : #diametre = 0.3
                        #print("center_ij :", center_ij)
                        # CORRECTION : r_vec doit pointer de i VERS j (attraction)
                        r_vec = center_j - positions[i]
                        #print("r_vec :", r_vec)
                    else :
                        r_vec = positions[j] - positions[i]
                    
                    distance = np.linalg.norm(r_vec)
                    #print("distance :", distance)
                    
                    if distance > 0:  # Éviter division par zéro
                        # Force gravitationnelle: F = G * m_i * m_j / r²
                        # Accélération: a = F / m_i = G * m_j / r² * (r_vec / r)
                        # CORRECTION : utiliser masses[j] (masse du corps attracteur) et non masses[i]
                        acceleration_magnitude = G * masses[j] / (distance**2)
                        acceleration_direction = r_vec / distance
                        accelerations[i] += acceleration_magnitude * acceleration_direction
        
        return accelerations 


def mise_a_jour(n_corps, positions, vitesses, masses, gravity_centers, dt):
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
    #accelerations = calculer_accelerations_vectorisees_2(positions, masses)
    
    # Mise à jour vectorisée (pas de boucle!)
    nouvelles_vitesses = vitesses + accelerations * dt
    nouvelles_positions = positions + vitesses * dt + 0.5 * accelerations * dt**2
    
    return nouvelles_positions, nouvelles_vitesses

def mise_a_jour_2(n_corps, positions, vitesses, masses, grid_pos, gravity_centers, dt):
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
    #accelerations = calculer_accelerations_vectorisees(positions, masses)
    accelerations = calculer_accelerations_vectorisees_2(n_corps, positions, masses, grid_pos, gravity_centers)
    
    # Mise à jour vectorisée (pas de boucle!)
    nouvelles_vitesses = vitesses + accelerations * dt
    nouvelles_positions = positions + vitesses * dt + 0.5 * accelerations * dt**2

    #print("nouvelles positions :", nouvelles_positions.shape)

    nouvelles_grid_pos = np.apply_along_axis(belongs_to, 1, nouvelles_positions)
    #print("nouvelles grid_pos :", nouvelles_grid_pos.shape)

  # bel_vec = np.vectorize(belongs_to)
  # nouvelles_grid_pos = bel_vec(positions)
    nouveaux_gravity_centers = update_gravity_centers(n_corps, nouvelles_positions, nouvelles_grid_pos, gravity_centers)
    
    return nouvelles_positions, nouvelles_vitesses, nouvelles_grid_pos, nouveaux_gravity_centers

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
    positions, vitesses, masses, couleurs, grid_pos, gravity_centers = creer_galaxie(n_corps)
    
    # Ajout du trou noir
    positions, vitesses, masses, couleurs, grid_pos = ajouter_trou_noir(
        positions, vitesses, masses, grid_pos, couleurs,
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
        positions, vitesses, masses, couleurs, grid_pos = ajouter_etoile(
            positions, vitesses, masses,grid_pos, couleurs,
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

        #print("grid_pos :", grid_pos)
        
        def updater(dt):
            nonlocal positions, vitesses, frame_count, grid_pos, gravity_centers
            frame_count += 1
            
            # AUGMENTEZ CE FACTEUR POUR VOIR LE MOUVEMENT !
            facteur_temps = 500  # Au lieu de 10

            #print("grid_pos :", grid_pos)

            dt = 0.1
            
            #positions, vitesses = mise_a_jour(positions, vitesses, masses, dt * facteur_temps)
            positions, vitesses, grid_pos, gravity_centers = mise_a_jour_2(n_corps, positions, vitesses, masses, grid_pos, gravity_centers, dt * facteur_temps)

            print("posiotions[10] :", positions[10])
            
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

def _init_galaxie(n_etoiles):
    """Initialise une galaxie pour le benchmark"""
    n_corps = n_etoiles + 1
    positions, vitesses, masses, couleurs, grid_pos, gravity_centers = creer_galaxie(n_corps)
    positions, vitesses, masses, couleurs, grid_pos = ajouter_trou_noir(
        positions, vitesses, masses, grid_pos, couleurs, 1e6, np.zeros(3)
    )
    for i in range(1, n_corps):
        r = np.random.uniform(1, 5)
        theta = np.random.uniform(0, 2*np.pi)
        pos = np.array([r*np.cos(theta), r*np.sin(theta), 0])
        masse = np.random.uniform(0.5, 10)
        positions, vitesses, masses, couleurs, grid_pos = ajouter_etoile(
            positions, vitesses, masses, grid_pos, couleurs, i, masse, pos
        )
    return n_corps, positions, vitesses, masses, couleurs, grid_pos, gravity_centers


def benchmark_vectorise(n_etoiles=100, n_iterations=50, dt=0.01):
    """
    Benchmark de la version vectorisée
    """
    n_corps, positions, vitesses, masses, couleurs, grid_pos, gravity_centers = _init_galaxie(n_etoiles)

    # Mesure du temps
    start = time.time()
    for _ in range(n_iterations):
        positions, vitesses, grid_pos, gravity_centers = mise_a_jour_2(n_corps, positions, vitesses, masses, grid_pos, gravity_centers, dt)
    end = time.time()
    
    return end - start


def benchmark_comparaison(n_etoiles_list=None, n_iterations=30, dt=0.01):
    """
    Compare les temps de calcul entre la version 1 (classes) et la version 2 (vectorisée)
    pour différents nombres d'étoiles.
    """
    if n_etoiles_list is None:
        n_etoiles_list = [50, 100, 200, 500]

    # Import de la fonction de benchmark v1
    try:
        from Corps_accel import NCorps, Corps
        def benchmark_v1(n_etoiles, n_iterations, dt):
            galaxy = NCorps()
            black_hole = Corps(mass=1e6, position=np.zeros(3))
            galaxy.add(black_hole)
            for _ in range(n_etoiles):
                r = np.random.uniform(1, 5)
                theta = np.random.uniform(0, 2*np.pi)
                pos = np.array([r*np.cos(theta), r*np.sin(theta), 0])
                masse = np.random.uniform(0.5, 10)
                star = Corps(mass=masse, position=pos)
                galaxy.add(star)
            start = time.time()
            for _ in range(n_iterations):
                galaxy.update(dt)
            return time.time() - start
        v1_disponible = True
    except ImportError:
        v1_disponible = False
        print("⚠️  Corps_accel.py introuvable : comparaison v1 désactivée\n")

    print("="*60)
    print("COMPARAISON TEMPS DE CALCUL : V1 (classes) vs V2 (vectorisée)")
    print(f"Nombre d'itérations : {n_iterations}")
    print("="*60)

    if v1_disponible:
        print(f"{'N étoiles':<12} {'V1 classes (s)':<18} {'V2 vectorisée (s)':<20} {'Speedup'}")
        print("-"*60)
    else:
        print(f"{'N étoiles':<12} {'V2 vectorisée (s)'}")
        print("-"*30)

    for n in n_etoiles_list:
        t2 = benchmark_vectorise(n_etoiles=n, n_iterations=n_iterations, dt=dt)
        if v1_disponible:
            t1 = benchmark_v1(n_etoiles=n, n_iterations=n_iterations, dt=dt)
            speedup = t1 / t2 if t2 > 0 else float('inf')
            print(f"{n:<12} {t1:<18.4f} {t2:<20.4f} x{speedup:.1f}")
        else:
            print(f"{n:<12} {t2:.4f}")

    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        t = benchmark_vectorise(n_etoiles=n, n_iterations=30)
        print(f"Temps pour {n} étoiles: {t:.4f} secondes")

    elif len(sys.argv) > 1 and sys.argv[1] == "--comparaison":
        benchmark_comparaison()
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--visualize":
        visualisation_vectorisee()
    
    else:
        print("Usage:")
        print("  python3 Corps_vectorise.py --benchmark [n_etoiles]")
        print("  python3 Corps_vectorise.py --comparaison")
        print("  python3 Corps_vectorise.py --visualize")