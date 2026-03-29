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
        self.n = n_corps
        self.positions = np.zeros((n_corps, 3), dtype=np.float64)
        self.vitesses = np.zeros((n_corps, 3), dtype=np.float64)
        self.masses = np.zeros(n_corps, dtype=np.float64)
        self.couleurs = np.zeros((n_corps, 3), dtype=np.float32)
        self.trou_noir_idx = 0
    
    def ajouter_trou_noir(self, masse, position, vitesse=None):
        if vitesse is None:
            vitesse = np.zeros(3)
        self.masses[self.trou_noir_idx] = masse
        self.positions[self.trou_noir_idx] = position
        self.vitesses[self.trou_noir_idx] = vitesse
        self.couleurs[self.trou_noir_idx] = np.array([0, 0, 0], dtype=np.float32)
    
    def ajouter_etoile(self, idx, masse, position, vitesse=None):
        if vitesse is None:
            vitesse = np.zeros(3)
        self.masses[idx] = masse
        self.positions[idx] = position
        self.vitesses[idx] = vitesse
        self.couleurs[idx] = self._couleur_selon_masse(masse)
    
    def _couleur_selon_masse(self, masse):
        if masse > 5:
            return np.array([150, 180, 255], dtype=np.float32)
        elif masse > 2:
            return np.array([255, 255, 255], dtype=np.float32)
        elif masse >= 1:  # CORRECTION : >= au lieu de >
            return np.array([255, 255, 200], dtype=np.float32)
        else:
            return np.array([250, 150, 100], dtype=np.float32)
    
    def calculer_accelerations_python(self):
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
        return _calcul_accelerations_numba_serial(
            self.positions, self.masses, G, self.n
        )
    
    def calculer_accelerations_numba_parallel(self):
        return _calcul_accelerations_numba_parallel(
            self.positions, self.masses, G, self.n
        )
    
    def update(self, dt, mode='numba_parallel'):
        if mode == 'python':
            accelerations = self.calculer_accelerations_python()
        elif mode == 'numba_serial':
            accelerations = self.calculer_accelerations_numba_serial()
        else:  # numba_parallel
            accelerations = self.calculer_accelerations_numba_parallel()
        
        # CORRECTION : positions mises à jour avec v(t) AVANT de modifier les vitesses
        self.positions += self.vitesses * dt + 0.5 * accelerations * dt**2
        self.vitesses  += accelerations * dt
    
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

@njit
def _calcul_accelerations_numba_serial(positions, masses, G, n):
    accelerations = np.zeros((n, 3), dtype=np.float64)
    
    for i in range(n):
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


@njit(parallel=True)
def _calcul_accelerations_numba_parallel(positions, masses, G, n):
    accelerations = np.zeros((n, 3), dtype=np.float64)
    
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

def creer_galaxie_test(n_etoiles):
    n_corps = n_etoiles + 1
    galaxie = GalaxieNumba(n_corps)
    galaxie.ajouter_trou_noir(1e6, np.zeros(3))
    
    for i in range(1, n_corps):
        r = np.random.uniform(2, 6)
        theta = np.random.uniform(0, 2*np.pi)
        pos = np.array([
            r * np.cos(theta),
            r * np.sin(theta),
            np.random.uniform(-0.3, 0.3)
        ])
        v = np.sqrt(G * 1e6 / r)
        vitesse = np.array([
            -v * np.sin(theta),
            v * np.cos(theta),
            0
        ])
        masse = np.random.uniform(0.5, 10)
        galaxie.ajouter_etoile(i, masse, pos, vitesse)
    
    return galaxie


def benchmark_numba(n_etoiles=100, n_iterations=30, dt=0.01):
    import multiprocessing
    
    n_coeurs = multiprocessing.cpu_count()
    # CORRECTION : gestion propre de l'import psutil
    try:
        import psutil
        n_coeurs_logiques = psutil.cpu_count(logical=True)
    except ImportError:
        n_coeurs_logiques = n_coeurs
    
    print("\n" + "="*80)
    print(f"BENCHMARK NUMBA - {n_etoiles} étoiles, {n_iterations} itérations")
    print(f"Plateforme: {n_coeurs} cœurs physiques, {n_coeurs_logiques} cœurs logiques")
    print("="*80)
    
    # 1. Version pure Python
    print("\n🔵 Version pure Python...")
    galaxie_py = creer_galaxie_test(n_etoiles)
    start = time.time()
    for _ in range(n_iterations):
        galaxie_py.update(dt, mode='python')
    t_python = time.time() - start
    print(f"   Temps: {t_python:.4f} s")
    
    # 2. Version Numba séquentielle
    print("\n🟢 Version Numba (séquentielle)...")
    galaxie_ns = creer_galaxie_test(n_etoiles)
    galaxie_ns.update(dt, mode='numba_serial')  # Warmup JIT
    start = time.time()
    for _ in range(n_iterations):
        galaxie_ns.update(dt, mode='numba_serial')
    t_numba_serial = time.time() - start
    print(f"   Temps: {t_numba_serial:.4f} s")
    
    # 3. Version Numba parallèle
    print("\n🟠 Version Numba (parallèle)...")
    galaxie_np = creer_galaxie_test(n_etoiles)
    galaxie_np.update(dt, mode='numba_parallel')  # Warmup JIT
    start = time.time()
    for _ in range(n_iterations):
        galaxie_np.update(dt, mode='numba_parallel')
    t_numba_parallel = time.time() - start
    print(f"   Temps: {t_numba_parallel:.4f} s")
    
    print("\n" + "-"*80)
    print("RÉSULTATS:")
    print(f"  • Python vs Numba serial:     {t_python/t_numba_serial:6.2f}x plus rapide")
    print(f"  • Python vs Numba parallel:   {t_python/t_numba_parallel:6.2f}x plus rapide")
    print(f"  • Numba parallel vs serial:   {t_numba_serial/t_numba_parallel:6.2f}x (gain parallélisation)")
    print(f"  • Efficacité parallèle:       {t_numba_serial/t_numba_parallel/n_coeurs*100:5.1f}%")
    print("="*80)
    
    return {
        'python': t_python,
        'numba_serial': t_numba_serial,
        'numba_parallel': t_numba_parallel,
        'n_coeurs': n_coeurs
    }


def benchmark_multi_taille():
    tailles = [50, 100, 200, 300, 400, 500]
    resultats = []
    
    print("\n" + "="*90)
    print("BENCHMARK MULTI-TAILLES AVEC NUMBA")
    print("="*90)
    print(f"{'Étoiles':>8} | {'Python (s)':>12} | {'Numba S (s)':>12} | {'Numba P (s)':>12} | {'Accélération':>12} | {'Gain parallèle':>12}")
    print("-"*90)
    
    for n in tailles:
        print(f"Test avec {n:3d} étoiles...", end='', flush=True)
        
        galaxie = creer_galaxie_test(n)
        start = time.time()
        for _ in range(10):
            galaxie.update(0.01, mode='python')
        t_py = time.time() - start
        
        galaxie = creer_galaxie_test(n)
        galaxie.update(0.01, mode='numba_serial')  # Warmup
        start = time.time()
        for _ in range(10):
            galaxie.update(0.01, mode='numba_serial')
        t_ns = time.time() - start
        
        galaxie = creer_galaxie_test(n)
        galaxie.update(0.01, mode='numba_parallel')  # Warmup
        start = time.time()
        for _ in range(10):
            galaxie.update(0.01, mode='numba_parallel')
        t_np = time.time() - start
        
        accel = t_py / t_np
        gain_par = t_ns / t_np
        
        print(f"\r{n:8d} | {t_py:12.4f} | {t_ns:12.4f} | {t_np:12.4f} | {accel:12.2f}x | {gain_par:12.2f}x")
        resultats.append((n, t_py, t_ns, t_np, accel, gain_par))
    
    print("="*90)
    print("\n📊 ANALYSE:")
    print("   - L'accélération augmente avec le nombre d'étoiles")
    print("   - Le gain de la parallélisation est limité par les cœurs disponibles")
    print("   - Numba est particulièrement efficace pour les grands N")
    
    return resultats


def visualisation_numba(n_etoiles=200, facteur_temps=200):
    from galaxy_generator import generate_galaxy
    import os

    galaxy_file = f"data/galaxy_{n_etoiles}"
    if not os.path.exists(galaxy_file):
        print(f"Génération du fichier {galaxy_file}...")
        generate_galaxy(n_stars=n_etoiles, output_file=galaxy_file)

    masses_list, pos_list, vel_list, _ = [], [], [], []
    with open(galaxy_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            masses_list.append(float(parts[0]))
            pos_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
            vel_list.append([float(parts[4]), float(parts[5]), float(parts[6])])

    n_corps = len(masses_list)
    galaxie = GalaxieNumba(n_corps)

    for i in range(n_corps):
        galaxie.masses[i] = masses_list[i]
        galaxie.positions[i] = pos_list[i]
        galaxie.vitesses[i] = vel_list[i]
        if i == 0:
            galaxie.couleurs[i] = np.array([0, 0, 0], dtype=np.float32)
        else:
            galaxie.couleurs[i] = galaxie._couleur_selon_masse(masses_list[i])

    print(f"Galaxie chargée depuis {galaxy_file} : {n_corps} corps")
    print("Préparation de la visualisation...")

    try:
        from visualizer3d_sans_vbo import Visualizer3D

        points_init = galaxie.get_points()
        visualizer = Visualizer3D(
            points=points_init,
            colors=galaxie.get_couleurs(),
            luminosities=galaxie.get_luminosites(),
            bounds=((-2, 2), (-2, 2), (-0.5, 0.5))
        )
        visualizer.camera_distance = 4
        frame_count = 0
        start_time = time.time()

        def updater_numba(dt_visu):
            nonlocal frame_count, start_time
            frame_count += 1
            dt_effectif = 0.001 * facteur_temps
            galaxie.update(dt_effectif, mode='numba_parallel')
            points = galaxie.get_points()

            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                distances = np.linalg.norm(points[1:], axis=1)
                print(f"Frame {frame_count:4d} | Temps réel: {elapsed:5.1f}s | "
                      f"Max distance: {distances.max():6.4f} ly")

            return points, galaxie.get_couleurs(), galaxie.get_luminosites()

        print(f"\n{'='*60}")
        print(f"VISUALISATION NUMBA — {n_corps} corps | Facteur: {facteur_temps}x")
        print(f"Appuyez sur ECHAP pour quitter")
        print(f"{'='*60}\n")

        visualizer.run_with_updater(updater_numba, dt=0.016)

    except ImportError:
        print("Erreur: module visualizer3d_sans_vbo non trouvé")
    except Exception as e:
        print(f"Erreur de visualisation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--benchmark":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            benchmark_numba(n_etoiles=n, n_iterations=30)
        
        elif sys.argv[1] == "--multi":
            benchmark_multi_taille()
        
        elif sys.argv[1] == "--visualize":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
            facteur = int(sys.argv[3]) if len(sys.argv) > 3 else 200
            visualisation_numba(n_etoiles=n, facteur_temps=facteur)
        
        else:
            print("Usage:")
            print("  python3 Corps_numba.py --benchmark [n_etoiles]")
            print("  python3 Corps_numba.py --multi")
            print("  python3 Corps_numba.py --visualize [n_etoiles] [facteur_temps]")
            print("\nExemples:")
            print("  python3 Corps_numba.py --visualize 200 200")
            print("  python3 Corps_numba.py --benchmark 500")
    
    else:
        print("Test rapide avec 100 étoiles...")
        benchmark_numba(n_etoiles=100, n_iterations=20)