#!/usr/bin/env python3
"""
VERSION 2 - VECTORISÉE AVEC NUMPY
Problème à N corps avec calculs vectorisés (NumPy)
Version stable et optimisée sans bugs
"""

import numpy as np
import time

# Constante gravitationnelle
G = 1.560339e-13

# ============================================================
# FONCTIONS DE CRÉATION ET GESTION DES DONNÉES
# ============================================================

def creer_galaxie(n_corps):
    positions = np.zeros((n_corps, 3), dtype=np.float64)
    vitesses = np.zeros((n_corps, 3), dtype=np.float64)
    masses = np.zeros(n_corps, dtype=np.float64)
    couleurs = np.zeros((n_corps, 3), dtype=np.float32)
    return positions, vitesses, masses, couleurs


def ajouter_trou_noir(positions, vitesses, masses, couleurs, masse, position, vitesse=None):
    if vitesse is None:
        vitesse = np.zeros(3)
    masses[0] = masse
    positions[0] = position
    vitesses[0] = vitesse
    couleurs[0] = np.array([0, 0, 0], dtype=np.float32)
    return positions, vitesses, masses, couleurs


def ajouter_etoile(positions, vitesses, masses, couleurs, idx, masse, position, vitesse=None):
    if vitesse is None:
        vitesse = np.zeros(3)
    masses[idx] = masse
    positions[idx] = position
    vitesses[idx] = vitesse
    couleurs[idx] = _couleur_selon_masse(masse)
    return positions, vitesses, masses, couleurs


def _couleur_selon_masse(masse):
    if masse > 5:
        return np.array([150, 180, 255], dtype=np.float32)
    elif masse > 2:
        return np.array([255, 255, 255], dtype=np.float32)
    elif masse >= 1:  # >= 1 comme indiqué dans le sujet
        return np.array([255, 255, 200], dtype=np.float32)
    else:
        return np.array([250, 150, 100], dtype=np.float32)


# ============================================================
# FONCTIONS DE CALCUL VECTORISÉ
# ============================================================

def calculer_accelerations_vectorisees(positions, masses):
    n = len(masses)
    pos_i = positions[:, np.newaxis, :]  # (n, 1, 3)
    pos_j = positions[np.newaxis, :, :]  # (1, n, 3)
    r_vectors = pos_j - pos_i            # (n, n, 3)
    distances = np.linalg.norm(r_vectors, axis=2) + 1e-10  # (n, n)
    m_sur_r3 = masses[np.newaxis, :] / (distances**3)
    np.fill_diagonal(m_sur_r3, 0)
    accelerations = G * np.sum(
        m_sur_r3[:, :, np.newaxis] * r_vectors,
        axis=1
    )
    return accelerations


def mise_a_jour(positions, vitesses, masses, dt):
    accelerations = calculer_accelerations_vectorisees(positions, masses)
    nouvelles_vitesses = vitesses + accelerations * dt
    nouvelles_positions = positions + vitesses * dt + 0.5 * accelerations * dt**2
    return nouvelles_positions, nouvelles_vitesses


# ============================================================
# FONCTIONS DE GESTION DES DONNÉES POUR VISUALISATION
# ============================================================

def get_points(positions):
    return positions.astype(np.float32)


def get_couleurs(couleurs):
    return couleurs.astype(np.float32)


def get_luminosites(masses):
    max_masse = masses.max()
    if max_masse > 0:
        return np.clip(masses / max_masse, 0.3, 1.0).astype(np.float32)
    return np.ones(len(masses), dtype=np.float32) * 0.5


# ============================================================
# FONCTION PRINCIPALE DE VISUALISATION
# ============================================================

def visualisation_vectorisee(n_etoiles=100, facteur_temps=500, dt_simulation=0.1):
    n_corps = n_etoiles + 1

    print("\n" + "="*60)
    print("VERSION 2 - VECTORISÉE AVEC NUMPY")
    print("="*60)

    positions, vitesses, masses, couleurs = creer_galaxie(n_corps)
    positions, vitesses, masses, couleurs = ajouter_trou_noir(
        positions, vitesses, masses, couleurs, masse=1e6, position=np.zeros(3)
    )

    print(f"Création de {n_etoiles} étoiles...")
    for i in range(1, n_corps):
        r = np.random.uniform(2, 8)
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(-0.2, 0.2)
        pos = np.array([
            r * np.cos(theta) * np.cos(phi),
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(phi) * 0.3
        ])
        v_orbital = np.sqrt(G * masses[0] / r)
        vitesse = np.array([-v_orbital * np.sin(theta), v_orbital * np.cos(theta), 0])
        masse = np.random.uniform(0.3, 12)
        positions, vitesses, masses, couleurs = ajouter_etoile(
            positions, vitesses, masses, couleurs, i, masse, pos, vitesse
        )

    print(f"Vitesse orbitale typique: {np.linalg.norm(vitesses[1]):.2e} ly/an")
    print(f"Masse totale: {masses.sum():.2e} masses solaires")

    points_init = get_points(positions)
    lum = get_luminosites(masses)
    bounds = ((-12, 12), (-12, 12), (-3, 3))

    print(f"Nombre de corps: {n_corps}")
    print(f"Positions initiales - min: {points_init.min(axis=0)}, max: {points_init.max(axis=0)}")

    try:
        from visualizer3d_sans_vbo import Visualizer3D

        visualizer = Visualizer3D(
            points=points_init,
            colors=couleurs,
            luminosities=lum,
            bounds=bounds
        )
        visualizer.camera_distance = 18

        frame_count = 0
        start_time = time.time()

        def updater(dt):
            nonlocal positions, vitesses, frame_count, start_time
            frame_count += 1
            dt_effectif = dt_simulation * facteur_temps
            positions, vitesses = mise_a_jour(positions, vitesses, masses, dt_effectif)

            distances = np.linalg.norm(positions[1:], axis=1)
            max_dist = np.max(distances)

            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                print(f"Frame {frame_count} | Temps écoulé: {elapsed:.1f}s | "
                      f"Max distance: {max_dist:.2f} ly | "
                      f"Étoile 1: ({positions[1,0]:.2f}, {positions[1,1]:.2f}, {positions[1,2]:.2f})")

            return (get_points(positions), get_couleurs(couleurs), get_luminosites(masses))

        print(f"\n🎮 Visualisation avec version vectorisée")
        print(f"   Facteur de temps: {facteur_temps}x")
        print(f"   Pas de temps effectif: {dt_simulation * facteur_temps:.2f} ans")
        print("   Appuyez sur ESC pour quitter\n")

        visualizer.run_with_updater(updater, dt=0.016)

    except ImportError as e:
        print(f"Erreur: module visualizer3d_sans_vbo non trouvé")
        print(f"Erreur détaillée: {e}")
    except Exception as e:
        print(f"Erreur de visualisation: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# FONCTION DE BENCHMARK
# ============================================================

def _init_galaxie_benchmark(n_etoiles):
    n_corps = n_etoiles + 1
    positions, vitesses, masses, couleurs = creer_galaxie(n_corps)
    positions, vitesses, masses, couleurs = ajouter_trou_noir(
        positions, vitesses, masses, couleurs, 1e6, np.zeros(3)
    )
    for i in range(1, n_corps):
        r = np.random.uniform(2, 6)
        theta = np.random.uniform(0, 2*np.pi)
        pos = np.array([r*np.cos(theta), r*np.sin(theta), np.random.uniform(-0.2, 0.2)])
        v_orbital = np.sqrt(G * 1e6 / r)
        vitesse = np.array([-v_orbital * np.sin(theta), v_orbital * np.cos(theta), 0])
        masse = np.random.uniform(0.5, 10)
        positions, vitesses, masses, couleurs = ajouter_etoile(
            positions, vitesses, masses, couleurs, i, masse, pos, vitesse
        )
    return positions, vitesses, masses


def benchmark_vectorise(n_etoiles=100, n_iterations=50, dt=0.01):
    positions, vitesses, masses = _init_galaxie_benchmark(n_etoiles)

    # CORRECTION : pas de warmup, NumPy n'a pas de JIT à compiler
    start = time.time()
    for _ in range(n_iterations):
        positions, vitesses = mise_a_jour(positions, vitesses, masses, dt)
    end = time.time()

    return end - start


def benchmark_comparaison(n_etoiles_list=None, n_iterations=30, dt=0.01):
    if n_etoiles_list is None:
        n_etoiles_list = [50, 100, 200, 500, 1000]

    try:
        from Corps_accel import NCorps, Corps

        def benchmark_v1(n_etoiles, n_iterations, dt):
            galaxy = NCorps()
            black_hole = Corps(mass=1e6, position=np.zeros(3))
            galaxy.add(black_hole)
            for _ in range(n_etoiles):
                r = np.random.uniform(2, 6)
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
        print("✅ Version 1 (classes) trouvée")

    except ImportError:
        v1_disponible = False
        print("⚠️  Corps_accel.py introuvable : comparaison v1 désactivée\n")

    print("\n" + "="*80)
    print("COMPARAISON DES PERFORMANCES: Version 1 (classes) vs Version 2 (vectorisée)")
    print(f"Nombre d'itérations par test: {n_iterations}")
    print("="*80)

    if v1_disponible:
        print(f"{'N étoiles':<12} {'V1 classes (s)':<18} {'V2 vectorisée (s)':<20} {'Speedup':<10} {'Complexité'}")
        print("-"*80)
    else:
        print(f"{'N étoiles':<12} {'V2 vectorisée (s)':<20} {'Complexité'}")
        print("-"*50)

    for n in n_etoiles_list:
        print(f"Test avec {n} étoiles...", end=" ", flush=True)
        t2 = benchmark_vectorise(n_etoiles=n, n_iterations=n_iterations, dt=dt)

        if v1_disponible:
            t1 = benchmark_v1(n_etoiles=n, n_iterations=n_iterations, dt=dt)
            speedup = t1 / t2 if t2 > 0 else float('inf')
            complexite = f"O({n}²)" if n <= 100 else f"~O({n}²)"
            print(f"\r{n:<12} {t1:<18.4f} {t2:<20.4f} x{speedup:>7.1f}   {complexite}")
        else:
            complexite = f"O({n}²)"
            print(f"\r{n:<12} {t2:<20.4f} {complexite}")

    print("="*80)

    if v1_disponible:
        print("\n📊 Analyse des résultats:")
        print("   - La version vectorisée est beaucoup plus rapide grâce à NumPy")
        print("   - L'accélération devient plus importante avec N élevé")
        print("   - La complexité reste O(N²) mais avec un facteur constant bien plus faible")


# ============================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--benchmark":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            print(f"\n🏃 Benchmark avec {n} étoiles...")
            t = benchmark_vectorise(n_etoiles=n, n_iterations=50)
            print(f"Temps total: {t:.4f} secondes")
            print(f"Temps par itération: {t/50:.4f} secondes")
            print(f"Opérations: ~{n**2 * 50 * 3 * 2 / 1e9:.1f} milliards d'opérations flottantes")

        elif sys.argv[1] == "--comparaison":
            benchmark_comparaison()

        elif sys.argv[1] == "--visualize":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            facteur = int(sys.argv[3]) if len(sys.argv) > 3 else 500
            visualisation_vectorisee(n_etoiles=n, facteur_temps=facteur)

        else:
            print("Usage:")
            print("  python3 Corps_vectorise.py --benchmark [n_etoiles]")
            print("  python3 Corps_vectorise.py --comparaison")
            print("  python3 Corps_vectorise.py --visualize [n_etoiles] [facteur_temps]")

    else:
        print("="*60)
        print("VERSION 2 - SIMULATION VECTORISÉE DE GALAXIE")
        print("="*60)
        print("\nCommandes disponibles:")
        print("  --visualize [n_etoiles] [facteur]  → Lance la visualisation")
        print("  --benchmark [n_etoiles]            → Teste les performances")
        print("  --comparaison                      → Compare V1 vs V2")
        print("-"*60)
        visualisation_vectorisee(n_etoiles=100, facteur_temps=500)