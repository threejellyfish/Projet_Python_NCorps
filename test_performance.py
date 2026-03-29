import numpy as np
import time
import os
import matplotlib.pyplot as plt
import signal

# --- IMPORTS ---
from Corps import NCorps as NCorpsNaive, Corps as CorpsNaive
from Corps_vectorise import mise_a_jour as mise_a_jour_vectorisee
from Corps_numba import _calcul_accelerations_numba_parallel as numba_parallel_accel
from Corps_BarnesHut import BarnesHutSim
from Corps_grid import NCorps as NCorpsGrid, Corps as CorpsGrid
from galaxy_generator import generate_galaxy


def benchmark():

    n_values = [100, 1000, 10000]
    iterations = 10
    dt = 0.01
    G = 1.560339e-13

    print(f"{'N':<10} | {'Méthode':<20} | {'Temps Moyen (s)':<15} | {'FPS':<10}")
    print("-" * 65)

    methods = ['Naïve (Objet)', 'Grille (Grid)', 'Vectorisée', 'Numba Parallel', 'Barnes-Hut']
    results = {m: [] for m in methods}

    for n in n_values:

        galaxy_file = f"data/galaxy_{n}"
        if not os.path.exists(galaxy_file):
            print(f"Génération du fichier {galaxy_file}...")
            generate_galaxy(n_stars=n, output_file=galaxy_file)

        masses, pos, vel = [], [], []
        with open(galaxy_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                masses.append(float(parts[0]))
                pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
                vel.append([float(parts[4]), float(parts[5]), float(parts[6])])

        masses = np.array(masses)
        pos = np.array(pos)
        vel = np.array(vel)

        # --- TIMEOUT HELPER ---
        def run_with_timeout(func):
            try:
                signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError))
                signal.alarm(60)
                t = func()
                signal.alarm(0)
                return t
            except TimeoutError:
                return '>60s'
            except Exception as e:
                print(f"Erreur: {e}")
                return None
            finally:
                signal.alarm(0)

        # --- NAIVE ---
        def run_naive():
            galaxy = NCorpsNaive()
            for i in range(len(masses)):
                galaxy.add(CorpsNaive(mass=masses[i], position=pos[i], speed=vel[i]))
            start = time.time()
            for _ in range(iterations):
                galaxy.update(dt)
            return (time.time() - start) / iterations

        results['Naïve (Objet)'].append(run_with_timeout(run_naive) if n <= 1000 else None)

        # --- GRID ---
        def run_grid():
            galaxy = NCorpsGrid(n=len(masses)-1, filename=galaxy_file)
            start = time.time()
            for _ in range(iterations):
                galaxy.update(dt)
            return (time.time() - start) / iterations

        results['Grille (Grid)'].append(run_with_timeout(run_grid) if n <= 10000 else None)

        # --- VECTORISE ---
        def run_vect():
            p_v, v_v = pos.copy(), vel.copy()
            gravity_centers = np.zeros((20, 20, 3), dtype=np.float32)
            start = time.time()
            for _ in range(iterations):
                p_v, v_v = mise_a_jour_vectorisee(len(masses), p_v, v_v, masses, gravity_centers, dt)
            return (time.time() - start) / iterations

        results['Vectorisée'].append(run_with_timeout(run_vect) if n <= 10000 else None)

        # --- NUMBA ---
        def run_numba():
            p_n, v_n = pos.copy(), vel.copy()
            _ = numba_parallel_accel(p_n, masses, G, len(masses))  # warmup
            start = time.time()
            for _ in range(iterations):
                acc = numba_parallel_accel(p_n, masses, G, len(masses))
                v_n += acc * dt
                p_n += v_n * dt
            return (time.time() - start) / iterations

        results['Numba Parallel'].append(run_with_timeout(run_numba) if n <= 10000 else None)

        # --- BARNES-HUT ---
        def run_bh():
            sim = BarnesHutSim(galaxy_file)
            sim.update(dt)
            start = time.time()
            for _ in range(iterations):
                sim.update(dt)
            return (time.time() - start) / iterations

        results['Barnes-Hut'].append(run_with_timeout(run_bh))

    # --- SAVE CSV ---
    import csv
    with open('perf_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['N'] + methods)
        for i, n in enumerate(n_values):
            row = [n]
            for m in methods:
                val = results[m][i]
                if isinstance(val, float):
                    row.append(f"{val:.6f}")
                elif val == '>60s':
                    row.append('>60s')
                else:
                    row.append('-')
            writer.writerow(row)

    # --- MAIN PLOT (LOG SCALE) ---
    plt.figure(figsize=(8, 6))
    for method in methods:
        y = [t if isinstance(t, float) else np.nan for t in results[method]]
        plt.plot(n_values, y, marker='o', label=method)

    plt.xlabel('Nombre de corps (N)')
    plt.ylabel('Temps moyen par itération (s)')
    plt.title('Benchmark N-corps (échelle logarithmique)')
    plt.legend()
    plt.grid(True, which='both')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('benchmark_ncorps_log.png')

    # --- FAST METHODS PLOT ---
    fast_methods = ['Vectorisée', 'Numba Parallel', 'Barnes-Hut']

    plt.figure(figsize=(8, 6))
    for method in fast_methods:
        y = [t if isinstance(t, float) else np.nan for t in results[method]]
        plt.plot(n_values, y, marker='o', label=method)

    plt.xlabel('Nombre de corps (N)')
    plt.ylabel('Temps moyen (s)')
    plt.title('Zoom sur les méthodes rapides')
    plt.legend()
    plt.grid(True, which='both')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('benchmark_fast_methods.png')

    print("Graphiques sauvegardés :")
    print("- benchmark_ncorps_log.png")
    print("- benchmark_fast_methods.png")
    print("- perf_results.csv")


if __name__ == "__main__":
    benchmark()