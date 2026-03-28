#!/usr/bin/env python3
"""
VERSION 3 - OPTIMISÉE AVEC NUMBA
Problème à N corps avec accélération JIT (Just-In-Time compilation)
"""

import numpy as np
import time
from numba import jit, njit, prange

G = 1.560339e-13  # Constante gravitationnelle

class GalaxieNumba:
    """
    Version optimisée avec Numba
    Les calculs critiques sont compilés en code machine
    """
    
    def __init__(self, n_corps):
        """
        Initialise une galaxie avec n_corps corps
        
        Args:
            n_corps (int): Nombre total de corps (étoiles + trou noir)
        """
        self.n = n_corps
        
        # Tableaux pour stocker toutes les données
        self.positions = np.zeros((n_corps, 3), dtype=np.float64)
        self.vitesses = np.zeros((n_corps, 3), dtype=np.float64)
        self.masses = np.zeros(n_corps, dtype=np.float64)
        self.couleurs = np.zeros((n_corps, 3), dtype=np.float32)
        self.grid_pos = np.zeros((n_corps, 2), dtype=np.int8)
        self.gravity_centers = np.zeros((20,20,3), dtype= np.float16) 
        
        self.trou_noir_idx = 0

    def belongs_to(self, pos):

        x_coor = pos[0]
        y_coor = pos[1]

        for i in range(0,20) :
            inf = -3 + i*( 3 + 3)/19
            sup = -3 + (i+1)*( 3 + 3)/19
            if x_coor >= inf and x_coor <= sup :
                ix = i
            if y_coor >= inf and y_coor <= sup :
                iy = i

        grid_index = np.zeros((2))
        grid_index[0] = ix 
        grid_index[1] = iy
        return grid_index

    def update_gravity_centers(self):
        for i in range(20):
            for j in range(20):
                self.gravity_centers[i,j] = self.compute_gravity_center(i,j)
        return self.gravity_centers


    def compute_gravity_center(self,coord_x,coord_y): 

        loc_pos = []
        #print("grid_pos :", grid_pos.shape)
        for i in range(self.n):
            if int(self.grid_pos[i,0]) == coord_x and int(self.grid_pos[i,1]) == coord_y :

                loc_pos.append(self.positions[i])
        loc_pos = np.array(loc_pos)
        return np.mean(loc_pos, axis=0) if np.shape(loc_pos)[0] > 0 else np.zeros((3,))

    
    def ajouter_trou_noir(self, masse, position, vitesse=np.zeros(3)):
        """Ajoute le trou noir central"""
        self.masses[self.trou_noir_idx] = masse
        self.positions[self.trou_noir_idx] = position
        self.vitesses[self.trou_noir_idx] = vitesse
        self.couleurs[self.trou_noir_idx] = [0, 0, 0]
    
    def ajouter_etoile(self, idx, masse, position, vitesse=np.zeros(3)):
        """Ajoute une étoile"""
        self.masses[idx] = masse
        self.positions[idx] = position
        self.vitesses[idx] = vitesse
        self.couleurs[idx] = self._couleur_selon_masse(masse)
    
    def _couleur_selon_masse(self, masse):
        """Détermine la couleur selon la masse"""
        if masse > 5:
            return np.array([150, 180, 255], dtype=np.float32)
        elif masse > 2:
            return np.array([255, 255, 255], dtype=np.float32)
        elif masse > 1:
            return np.array([255, 255, 200], dtype=np.float32)
        else:
            return np.array([250, 150, 100], dtype=np.float32)
        
    def calculer_accelerations_vectorisees_2(self):
        "Grid"
        accelerations = np.zeros((self.n, 3), dtype=np.float64)
        gravity_centers = self.update_gravity_centers()

        for i in range(self.n):
            #print(" star id :", i)
            for j in range(self.n):
                if i != j:

                    center_j = gravity_centers[int(self.grid_pos[j,0]), int(self.grid_pos[j,1])]
                    if 1/2* np.linalg.norm(self.positions[i] - center_j) > 0.3 : #diametre = 0.3

                        r_vec = self.positions[i] - center_j
                    else :
                        r_vec = self.positions[j] - self.positions[i]
                    
            ax = ay = az = 0.0
            for j in range(n):
                if i != j:
                    center_j = gravity_centers[int(self.grid_pos[j,0]), int(self.grid_pos[j,1])]
                    dcx = center_j[j, 0] - self.positions[i, 0]
                    dcy = center_j[j, 1] - self.positions[i, 1]
                    dcz = center_j[j, 2] - self.positions[i, 2]
                    if 1/2* np.sqrt(dcx*dcx + dcy*dcy + dcz*dcz) + 1e-10 > 0.3 : #diametre = 0
                        r = np.sqrt(dcx*dcx + dcy*dcy + dcz*dcz) + 1e-10
                        
                    else :
                        dx = self.positions[j, 0] - self.positions[i, 0]
                        dy = self.positions[j, 1] - self.positions[i, 1]
                        dz = self.positions[j, 2] - self.positions[i, 2]
                    
                        r = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-10
                    
                    facteur = G * self.masses[j] / (r * r * r)
                    
                    ax += facteur * dx
                    ay += facteur * dy
                    az += facteur * dz
            
            accelerations[i, 0] = ax
            accelerations[i, 1] = ay
            accelerations[i, 2] = az
        
        return accelerations 
    
    def calculer_accelerations_python(self):
        """
        Version pure Python (sans optimisation)
        Pour comparer avec Numba
        """
        n = self.n
        accelerations = np.zeros((n, 3), dtype=np.float64)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = self.positions[j, 0] - self.positions[i, 0]
                    dy = self.positions[j, 1] - self.positions[i, 1]
                    dz = self.positions[j, 2] - self.positions[i, 2]
                    
                    r = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-10
                    
                    facteur = G * self.masses[j] / (r * r * r)
                    
                    accelerations[i, 0] += facteur * dx
                    accelerations[i, 1] += facteur * dy
                    accelerations[i, 2] += facteur * dz
        
        return accelerations
    
    def calculer_accelerations_numba_serial(self):
        """
        Version optimisée avec Numba (exécution séquentielle)
        Utilise @njit pour compiler la fonction
        """
        return _calcul_accelerations_numba_serial(
            self.positions, self.masses, G, self.n
        )
    
    def calculer_accelerations_numba_parallel(self):
        """
        Version optimisée avec Numba (parallélisée)
        Utilise @njit(parallel=True) et prange
        """
        return _calcul_accelerations_numba_parallel(
            self.positions, self.masses, G, self.n
        )
    
    def update(self, dt, mode='numba_parallel'):
        """
        Met à jour positions et vitesses
        
        Args:
            dt: pas de temps
            mode: 'python', 'numba_serial', ou 'numba_parallel'
        """
        if mode == 'python':
            accelerations = self.calculer_accelerations_python()
        elif mode == 'numba_serial':
            accelerations = self.calculer_accelerations_numba_serial()
        else:  # numba_parallel
            accelerations = self.calculer_accelerations_numba_parallel()
        
        # Mise à jour (vectorisée)
        self.vitesses += accelerations * dt
        self.positions += self.vitesses * dt + 0.5 * accelerations * dt**2
    
    def get_points(self):
        return self.positions.astype(np.float32)
    
    def get_couleurs(self):
        return self.couleurs.astype(np.float32)
    
    def get_masses(self):
        return self.masses.astype(np.float32)
    
    def get_luminosites(self):
        max_masse = self.masses.max()
        if max_masse > 0:
            return np.clip(self.masses / max_masse, 0.3, 1.0).astype(np.float32)
        return np.ones(self.n, dtype=np.float32) * 0.5


# ============================================================
# FONCTIONS OPTIMISÉES AVEC NUMBA (EN DEHORS DE LA CLASSE)
# ============================================================

@njit  # Compilation JIT sans parallélisation
def _calcul_accelerations_numba_serial(positions, masses, G, n):
    """
    Calcul des accélérations avec Numba (séquentiel)
    """
    accelerations = np.zeros((n, 3), dtype=np.float64)
    
    for i in range(n):
        ax = ay = az = 0.0
        for j in range(n):
            if i != j:
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                
                # Éviter la division par zéro
                r = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-10
                
                # Facteur = G * m_j / r^3
                facteur = G * masses[j] / (r * r * r)
                
                ax += facteur * dx
                ay += facteur * dy
                az += facteur * dz
        
        accelerations[i, 0] = ax
        accelerations[i, 1] = ay
        accelerations[i, 2] = az
    
    return accelerations

def _calculer_accelerations_numba_2(n_corps, positions, masses, grid_pos, gravity_centers):
        "Grid"
        accelerations = np.zeros((n_corps, 3), dtype=np.float64)
        #self.gravity_centers = np.ones((20,20,3), dtype=np.int16)
        gravity_centers = self.update_gravity_centers()
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
                        r_vec = positions[i] - center_j
                        #print("r_vec :", r_vec)
                    else :
                        r_vec = positions[j] - positions[i]
                    
                    # parallel=True + prange = parallélisation automatique
        for i in prange(n):
            ax = ay = az = 0.0
            for j in range(n):
                if i != j:
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dz = positions[j, 2] - positions[i, 2]
                    
                    r = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-10
                    
                    facteur = G * masses[j] / (r * r * r)
                    
                    ax += facteur * dx
                    ay += facteur * dy
                    az += facteur * dz
            
            accelerations[i, 0] = ax
            accelerations[i, 1] = ay
            accelerations[i, 2] = az
        
        return accelerations 


@njit(parallel=True)  # Compilation avec parallélisation
def _calcul_accelerations_numba_parallel(positions, masses, G, n):
    """
    Calcul des accélérations avec Numba (parallélisé)
    Utilise prange pour distribuer les itérations sur les cœurs CPU
    """
    accelerations = np.zeros((n, 3), dtype=np.float64)
    
    # parallel=True + prange = parallélisation automatique
    for i in prange(n):
        ax = ay = az = 0.0
        for j in range(n):
            if i != j:
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                
                r = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-10
                
                facteur = G * masses[j] / (r * r * r)
                
                ax += facteur * dx
                ay += facteur * dy
                az += facteur * dz
        
        accelerations[i, 0] = ax
        accelerations[i, 1] = ay
        accelerations[i, 2] = az
    
    return accelerations


# ============================================================
# FONCTIONS DE TEST ET BENCHMARK
# ============================================================

def benchmark_numba(n_etoiles=100, n_iterations=20, dt=0.01):
    """
    Compare les performances des différentes versions
    """
    import multiprocessing
    n_coeurs = multiprocessing.cpu_count()
    
    print("\n" + "="*70)
    print(f"BENCHMARK NUMBA - {n_etoiles} étoiles, {n_iterations} itérations")
    print(f"Plateforme: {n_coeurs} cœurs CPU disponibles")
    print("="*70)
    
    n_corps = n_etoiles + 1
    galaxie = GalaxieNumba(n_corps)
    
    # Initialisation
    galaxie.ajouter_trou_noir(1e6, np.zeros(3))
    for i in range(1, n_corps):
        r = np.random.uniform(1, 5)
        theta = np.random.uniform(0, 2*np.pi)
        pos = np.array([r*np.cos(theta), r*np.sin(theta), 0])
        masse = np.random.uniform(0.5, 10)
        galaxie.ajouter_etoile(i, masse, pos)
    
    # 1. Version pure Python
    print("\n🔵 Version pure Python...")
    start = time.time()
    for _ in range(n_iterations):
        galaxie.update(dt, mode='python')
    t_python = time.time() - start
    print(f"   Temps: {t_python:.4f} s")
    
    # 2. Version Numba séquentielle
    print("\n🟢 Version Numba (séquentielle)...")
    # Première exécution (compilation)
    galaxie.update(dt, mode='numba_serial')
    
    start = time.time()
    for _ in range(n_iterations):
        galaxie.update(dt, mode='numba_serial')
    t_numba_serial = time.time() - start
    print(f"   Temps: {t_numba_serial:.4f} s")
    
    # 3. Version Numba parallèle
    print("\n🟠 Version Numba (parallèle)...")
    # Première exécution (compilation)
    galaxie.update(dt, mode='numba_parallel')
    
    start = time.time()
    for _ in range(n_iterations):
        galaxie.update(dt, mode='numba_parallel')
    t_numba_parallel = time.time() - start
    print(f"   Temps: {t_numba_parallel:.4f} s")
    
    # Calcul des accélérations
    print("\n" + "-"*50)
    print("RÉSULTATS:")
    print(f"  • Python vs Numba serial: {t_python/t_numba_serial:.2f}x plus rapide")
    print(f"  • Python vs Numba parallel: {t_python/t_numba_parallel:.2f}x plus rapide")
    print(f"  • Numba parallel vs serial: {t_numba_serial/t_numba_parallel:.2f}x (gain parallélisation)")
    print(f"  • Efficacité parallèle: {t_numba_serial/t_numba_parallel/n_coeurs*100:.1f}%")
    
    return {
        'python': t_python,
        'numba_serial': t_numba_serial,
        'numba_parallel': t_numba_parallel,
        'n_coeurs': n_coeurs
    }


def benchmark_multi_taille():
    """
    Teste différentes tailles de galaxies
    """
    tailles = [50, 100, 200, 300, 400, 500]
    resultats = []
    
    print("\n" + "="*70)
    print("BENCHMARK MULTI-TAILLES AVEC NUMBA")
    print("="*70)
    print(f"{'Étoiles':>8} | {'Python (s)':>12} | {'Numba S (s)':>12} | {'Numba P (s)':>12} | {'Accélération':>12}")
    print("-"*70)
    
    for n in tailles:
        print(f"Test avec {n} étoiles...", end='', flush=True)
        
        # Création de la galaxie
        n_corps = n + 1
        galaxie = GalaxieNumba(n_corps)
        
        galaxie.ajouter_trou_noir(1e6, np.zeros(3))
        for i in range(1, n_corps):
            r = np.random.uniform(1, 5)
            theta = np.random.uniform(0, 2*np.pi)
            pos = np.array([r*np.cos(theta), r*np.sin(theta), 0])
            masse = np.random.uniform(0.5, 10)
            galaxie.ajouter_etoile(i, masse, pos)
        
        # Benchmark Python
        start = time.time()
        for _ in range(10):
            galaxie.update(0.01, mode='python')
        t_py = time.time() - start
        
        # Benchmark Numba serial
        galaxie.update(0.01, mode='numba_serial')  # Warmup
        start = time.time()
        for _ in range(10):
            galaxie.update(0.01, mode='numba_serial')
        t_ns = time.time() - start
        
        # Benchmark Numba parallel
        galaxie.update(0.01, mode='numba_parallel')  # Warmup
        start = time.time()
        for _ in range(10):
            galaxie.update(0.01, mode='numba_parallel')
        t_np = time.time() - start
        
        accel = t_py / t_np
        
        print(f"\r{n:8d} | {t_py:12.4f} | {t_ns:12.4f} | {t_np:12.4f} | {accel:12.2f}x")
        resultats.append((n, t_py, t_ns, t_np, accel))
    
    return resultats


def visualisation_numba():
    """
    Lance la visualisation avec la version Numba
    """
    import os
    
    n_etoiles = 50  # Plus petit pour commencer
    n_corps = n_etoiles + 1
    galaxie = GalaxieNumba(n_corps)
    
    # Création d'une galaxie
    galaxie.ajouter_trou_noir(1e6, np.zeros(3))
    
    print("Création des étoiles...")
    for i in range(1, n_corps):
        r = np.random.uniform(2, 5)
        theta = np.random.uniform(0, 2*np.pi)
        
        pos = np.array([
            r * np.cos(theta),
            r * np.sin(theta),
            np.random.uniform(-0.5, 0.5)
        ])
        
        # Vitesse orbitale pour que ça tourne
        v = np.sqrt(G * 1e6 / r) * 0.5  # Facteur 0.5 pour ralentir un peu
        vitesse = np.array([
            -v * np.sin(theta),
            v * np.cos(theta),
            0
        ])
        
        masse = np.random.uniform(0.5, 10)
        galaxie.ajouter_etoile(i, masse, pos, vitesse)
    
    print("Préparation de la visualisation...")
    
    # Visualisation
    try:
        from visualizer3d_sans_vbo import Visualizer3D
        
        points_init = galaxie.get_points()
        print(f"Points initiaux: min={points_init.min(axis=0)}, max={points_init.max(axis=0)}")
        
        visualizer = Visualizer3D(
            points=points_init,
            colors=galaxie.get_couleurs(),
            luminosities=galaxie.get_luminosites(),
            bounds=((-8, 8), (-8, 8), (-2, 2))
        )
        
        visualizer.camera_distance = 15
        
        # Compteur pour le débogage
        frame_count = 0
        
        def updater_numba(dt):
            nonlocal frame_count
            frame_count += 1
            
            # Met à jour la galaxie
            galaxie.update(dt, mode='numba_parallel')
            
            # Récupère les nouvelles positions
            points = galaxie.get_points()
            couleurs = galaxie.get_couleurs()
            luminosites = galaxie.get_luminosites()
            
            # Affiche toutes les 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: un point à {points[1, 0]:.2f}, {points[1, 1]:.2f}, {points[1, 2]:.2f}")
            
            return points, couleurs, luminosites
        
        print("\n🎮 Visualisation avec Numba - Appuyez sur ECHAP pour quitter")
        print(f"   {n_etoiles} étoiles, calcul parallélisé")
        print("   Les étoiles DEVRAIENT bouger...")
        
        visualizer.run_with_updater(updater_numba, dt=0.05)  # dt plus grand
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--benchmark":
            # Benchmark simple
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            benchmark_numba(n_etoiles=n, n_iterations=30)
        
        elif sys.argv[1] == "--multi":
            # Benchmark multi-tailles
            benchmark_multi_taille()
        
        elif sys.argv[1] == "--visualize":
            # Visualisation
            visualisation_numba()
        
        else:
            print("Usage:")
            print("  python3 Corps_numba.py --benchmark [n_etoiles]")
            print("  python3 Corps_numba.py --multi")
            print("  python3 Corps_numba.py --visualize")
    else:
        # Test rapide
        print("Test rapide de Numba...")
        res = benchmark_numba(n_etoiles=50, n_iterations=10)
        print("\nTest terminé!")