#!/usr/bin/env python3
"""
Étude de l'effet du pas de temps (dt) sur la stabilité de la simulation N-corps.
"""

import numpy as np
import matplotlib.pyplot as plt
from Corps_vectorise import calculer_accelerations_vectorisees, mise_a_jour

G = 1.560339e-13


def init_galaxie(n_etoiles, seed=42):
    """Initialise une galaxie reproductible (seed fixe pour comparaison équitable)."""
    np.random.seed(seed)
    n_corps = n_etoiles + 1

    positions = np.zeros((n_corps, 3), dtype=np.float64)
    vitesses  = np.zeros((n_corps, 3), dtype=np.float64)
    masses    = np.zeros(n_corps, dtype=np.float64)

    # Trou noir central
    masses[0] = 1e6

    for i in range(1, n_corps):
        r     = np.random.uniform(2, 6)
        theta = np.random.uniform(0, 2 * np.pi)
        phi   = np.random.uniform(-0.1, 0.1)

        positions[i] = [
            r * np.cos(theta) * np.cos(phi),
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(phi)
        ]

        v_orb = np.sqrt(G * masses[0] / r)
        vitesses[i] = [-v_orb * np.sin(theta), v_orb * np.cos(theta), 0]
        masses[i]   = np.random.uniform(0.5, 10)

    return positions, vitesses, masses


def energie_totale(positions, vitesses, masses):
    """Calcule l'énergie mécanique totale (cinétique + potentielle)."""
    # Énergie cinétique
    Ec = 0.5 * np.sum(masses * np.sum(vitesses**2, axis=1))

    # Énergie potentielle gravitationnelle
    Ep = 0.0
    n = len(masses)
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[j] - positions[i])
            if r > 0:
                Ep -= G * masses[i] * masses[j] / r

    return Ec + Ep


def etude_pas_de_temps(n_etoiles=50, n_iterations=300):
    dt_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]

    print("=" * 65)
    print("ÉTUDE DU PAS DE TEMPS")
    print(f"N étoiles: {n_etoiles} | Itérations: {n_iterations}")
    print("=" * 65)
    print(f"{'dt (ans)':<12} | {'E0':<14} | {'Ef':<14} | {'Dérive (%)':<12} | {'Stable'}")
    print("-" * 65)

    resultats = {}

    for dt in dt_values:
        positions, vitesses, masses = init_galaxie(n_etoiles)

        E0 = energie_totale(positions, vitesses, masses)
        energies = [E0]

        diverge = False
        for step in range(n_iterations):
            positions, vitesses = mise_a_jour(positions, vitesses, masses, dt)

            # Détection de divergence (étoiles qui s'échappent à l'infini)
            if np.any(np.abs(positions) > 1e6) or np.any(np.isnan(positions)):
                diverge = True
                break

            if step % 10 == 0:
                E = energie_totale(positions, vitesses, masses)
                energies.append(E)

        Ef = energies[-1]
        derive = (Ef - E0) / abs(E0) * 100 if E0 != 0 else float('inf')
        stable = "✓" if not diverge and abs(derive) < 10 else "✗"

        print(f"{dt:<12.3f} | {E0:<14.4e} | {Ef:<14.4e} | {derive:<12.2f} | {stable}")
        resultats[dt] = energies

    print("=" * 65)
    return resultats, dt_values


def tracer_resultats(resultats, dt_values, n_iterations):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Graphe 1 : évolution de l'énergie pour chaque dt ---
    ax1 = axes[0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(dt_values)))

    for dt, color in zip(dt_values, colors):
        energies = resultats[dt]
        steps = np.arange(len(energies)) * 10
        E0 = energies[0]
        derive_rel = [(E - E0) / abs(E0) * 100 for E in energies]
        ax1.plot(steps, derive_rel, label=f"dt = {dt} ans", color=color)

    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.set_xlabel("Itération")
    ax1.set_ylabel("Dérive de l'énergie (%)")
    ax1.set_title("Dérive de l'énergie mécanique selon dt")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-50, 200)

    # --- Graphe 2 : dérive finale vs dt ---
    ax2 = axes[1]
    derives_finales = []
    for dt in dt_values:
        energies = resultats[dt]
        E0 = energies[0]
        Ef = energies[-1]
        derives_finales.append(abs((Ef - E0) / abs(E0) * 100))

    ax2.loglog(dt_values, derives_finales, marker='o', color='steelblue', linewidth=2)
    ax2.set_xlabel("Pas de temps dt (années terrestres)")
    ax2.set_ylabel("|Dérive énergie| (%) — échelle log")
    ax2.set_title("Erreur finale vs pas de temps (échelle log-log)")
    ax2.grid(True, which='both', alpha=0.3)

    # Annotation zones stable/instable
    ax2.axvline(0.05, color='red', linestyle='--', alpha=0.5, label="Seuil instabilité")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("etude_pas_de_temps.png", dpi=150)
    print("\nGraphique sauvegardé : etude_pas_de_temps.png")
    plt.show()


if __name__ == "__main__":
    N_ETOILES   = 50
    N_ITERATIONS = 300

    resultats, dt_values = etude_pas_de_temps(
        n_etoiles=N_ETOILES,
        n_iterations=N_ITERATIONS
    )

    tracer_resultats(resultats, dt_values, N_ITERATIONS)

    print("""
INTERPRÉTATION :
- Petits dt (0.001–0.01 ans) : l'énergie se conserve bien → simulation stable.
- Grands dt (0.1–1.0 ans)    : l'énergie dérive fortement → la simulation diverge.

EXPLICATION :
  Le schéma d'intégration (Euler/Verlet) approxime la trajectoire réelle par
  des segments. L'erreur de troncature est proportionnelle à dt² par pas, et
  s'accumule sur N itérations. Si dt est trop grand, l'erreur dépasse la
  courbure réelle de l'orbite gravitationnelle : une étoile "saute" au-delà
  du point d'attraction et s'éloigne au lieu de rester en orbite.
  C'est l'instabilité numérique, indépendante du modèle physique.
""")
