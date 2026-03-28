import numpy as np
import time
import os

# --- IMPORTS ---
# 1. Version Naïve (Objet)
from Corps_accel import NCorps, Corps

# 2. Version Vectorisée (NumPy)
from Corps_vectorise import mise_a_jour as update_vectorized

# 3. Version Numba
from Corps_numba import _calcul_accelerations_numba_parallel as numba_parallel_accel

# 4. Version Barnes-Hut
from Corps_BarnesHut import BarnesHutSim

def benchmark():
    # Paramètres
    n_values = [100, 1000, 10000] # On évite 100 000 pour la Naïve/Vectorisée (trop lent/lourd)
    iterations = 10
    dt = 0.01
    G = 1.560339e-13

    print(f"{'N':<10} | {'Méthode':<20} | {'Temps Moyen (s)':<15} | {'FPS':<10}")
    print("-" * 65)

    for n in n_values:
        # --- PRÉPARATION DES DONNÉES COMMUNES ---
        pos = np.random.uniform(-5, 5, (n, 3)).astype(np.float64)
        vel = np.random.uniform(-0.1, 0.1, (n, 3)).astype(np.float64)
        masses = np.random.uniform(0.1, 10, n).astype(np.float64)

        # --- 1. TEST NAÏF (OBJET) ---
        if n <= 1000: # Trop lent au delà
            galaxy_naive = NCorps()
            for i in range(n):
                galaxy_naive.add(Corps(mass=masses[i], position=pos[i], speed=vel[i]))
            
            start = time.time()
            for _ in range(iterations):
                galaxy_naive.update(dt)
            t = (time.time() - start) / iterations
            print(f"{n:<10} | {'Naïve (Objet)':<20} | {t:<15.6f} | {1/t:<10.1f}")

        # --- 2. TEST VECTORISÉ (NUMPY) ---
        if n <= 10000: # Limite mémoire/vitesse O(N²)
            p_v, v_v = pos.copy(), vel.copy()
            start = time.time()
            for _ in range(iterations):
                p_v, v_v = update_vectorized(p_v, v_v, masses, dt)
            t = (time.time() - start) / iterations
            print(f"{n:<10} | {'Vectorisée':<20} | {t:<15.6f} | {1/t:<10.1f}")

        # --- 3. TEST NUMBA PARALLEL ---
        if n <= 10000:
            p_n, v_n = pos.copy(), vel.copy()
            # Warmup (compilation)
            _ = numba_parallel_accel(p_n, masses, G, n)
            
            start = time.time()
            for _ in range(iterations):
                acc = numba_parallel_accel(p_n, masses, G, n)
                v_n += acc * dt
                p_n += v_n * dt
            t = (time.time() - start) / iterations
            print(f"{n:<10} | {'Numba Parallel':<20} | {t:<15.6f} | {1/t:<10.1f}")

        # --- 4. TEST BARNES-HUT ---
        # Note: Barnes-Hut nécessite un fichier ou une structure chargée. 
        # Pour le test on peut simuler le chargement ou adapter la classe.
        # Ici on initialise une classe "factice" pour le test de performance
        try:
            # On crée un fichier temporaire pour que BarnesHutSim puisse charger
            temp_file = "temp_bench.txt"
            with open(temp_file, "w") as f:
                for i in range(n):
                    f.write(f"{masses[i]} {pos[i,0]} {pos[i,1]} {pos[i,2]} {vel[i,0]} {vel[i,1]} {vel[i,2]}\n")
            
            sim_bh = BarnesHutSim(temp_file)
            # Warmup
            sim_bh.update(dt)
            
            start = time.time()
            for _ in range(iterations):
                sim_bh.update(dt)
            t = (time.time() - start) / iterations
            print(f"{n:<10} | {'Barnes-Hut':<20} | {t:<15.6f} | {1/t:<10.1f}")
            os.remove(temp_file)
        except Exception as e:
            print(f"{n:<10} | {'Barnes-Hut':<20} | Erreur: {e}")

if __name__ == "__main__":
    benchmark()