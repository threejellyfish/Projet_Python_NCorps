import numpy as np
import os
from numba import njit, prange
#from visualizer3d_sans_vbo import Visualizer3D

# --- CONFIGURATION (Inspirée de votre version vectorisée) ---
G = 1.560339e-13
THETA = 0.6
EPS = 0.2          # Softening pour la stabilité
MAX_NODES = 200000
FACTEUR_TEMPS = 100 # Le facteur que vous utilisez dans votre version vectorisée

def load_galaxy_file(filename):
    """Lecture de votre format de fichier spécifique"""
    if not os.path.exists(filename):
        print(f"Fichier {filename} introuvable.")
        return None
    data = np.loadtxt(filename)
    if data.ndim == 1: data = data.reshape(1, -1)
    
    masses = data[:, 0].astype(np.float32)
    pos = data[:, 1:4].astype(np.float32)
    vel = data[:, 4:7].astype(np.float32)
    
    # Couleurs selon vos règles
    colors = np.zeros((len(masses), 3), dtype=np.float32)
    for i in range(len(masses)):
        m = masses[i]
        if i == 0: colors[i] = [255, 0, 0] # Trou noir en rouge pour le voir
        elif m > 5: colors[i] = [150, 180, 255]
        elif m > 2: colors[i] = [255, 255, 255]
        elif m >= 1: colors[i] = [255, 255, 200]
        else: colors[i] = [250, 150, 100]
    return pos, vel, masses, colors

@njit
def build_tree(pos, masses, p_min, p_max):
    num_particles = pos.shape[0]
    child_idx = np.full((MAX_NODES, 8), -1, dtype=np.int32)
    node_mass = np.zeros(MAX_NODES, dtype=np.float32)
    node_com = np.zeros((MAX_NODES, 3), dtype=np.float32)
    node_size = np.zeros(MAX_NODES, dtype=np.float32)
    next_node_idx = 1
    root_size = (p_max - p_min) + 0.01
    node_size[0] = root_size
    for i in range(num_particles):
        curr = 0
        sz = root_size
        cx, cy, cz = p_min + sz/2, p_min + sz/2, p_min + sz/2
        while True:
            old_m = node_mass[curr]
            new_m = old_m + masses[i]
            node_com[curr] = (node_com[curr] * old_m + pos[i] * masses[i]) / new_m
            node_mass[curr] = new_m
            octant = 0
            if pos[i, 0] > cx: octant += 1
            if pos[i, 1] > cy: octant += 2
            if pos[i, 2] > cz: octant += 4
            child = child_idx[curr, octant]
            if child == -1:
                child_idx[curr, octant] = next_node_idx
                node_size[next_node_idx] = sz / 2
                node_com[next_node_idx] = pos[i]
                node_mass[next_node_idx] = masses[i]
                next_node_idx += 1
                break
            else:
                curr = child
                sz /= 2
                cx += sz/2 if pos[i, 0] > cx else -sz/2
                cy += sz/2 if pos[i, 1] > cy else -sz/2
                cz += sz/2 if pos[i, 2] > cz else -sz/2
    return child_idx, node_mass, node_com, node_size

@njit(parallel=True)
def calculate_acceleration(pos, masses, child_idx, node_mass, node_com, node_size):
    n = pos.shape[0]
    acc = np.zeros((n, 3), dtype=np.float32)
    for i in prange(n):
        stack = np.zeros(64, dtype=np.int32)
        stack_ptr = 1
        stack[0] = 0
        p_i = pos[i]
        while stack_ptr > 0:
            stack_ptr -= 1
            curr = stack[stack_ptr]
            dx, dy, dz = node_com[curr, 0]-p_i[0], node_com[curr, 1]-p_i[1], node_com[curr, 2]-p_i[2]
            dist_sq = dx**2 + dy**2 + dz**2 + EPS**2
            dist = np.sqrt(dist_sq)
            if node_size[curr] / dist < THETA:
                f = G * node_mass[curr] / (dist_sq * dist)
                acc[i, 0] += f * dx; acc[i, 1] += f * dy; acc[i, 2] += f * dz
            else:
                has_child = False
                for o in range(8):
                    child = child_idx[curr, o]
                    if child != -1:
                        stack[stack_ptr] = child; stack_ptr += 1; has_child = True
                if not has_child and dist > 1e-6:
                    f = G * node_mass[curr] / (dist_sq * dist)
                    acc[i, 0] += f * dx; acc[i, 1] += f * dy; acc[i, 2] += f * dz
    return acc

class BarnesHutSim:
    def __init__(self, filename):
        res = load_galaxy_file(filename)
        self.pos, self.vel, self.masses, self.colors = res
        self.lum = np.clip(self.masses / (self.masses.max()+1e-5), 0.3, 1.0)

    def update(self, dt):
        # On calcule les limites de l'arbre
        p_min, p_max = np.min(self.pos), np.max(self.pos)
        
        # Barnes-Hut
        child_idx, node_mass, node_com, node_size = build_tree(self.pos, self.masses, p_min, p_max)
        acc = calculate_acceleration(self.pos, self.masses, child_idx, node_mass, node_com, node_size)
        
        # Intégration Newtonienne (exactement comme votre version vectorisée)
        self.pos += self.vel * dt + 0.5 * acc * (dt**2)
        self.vel += acc * dt
        return self.pos, self.colors, self.lum

if __name__ == "__main__":
    # Utilisez votre fichier ici
    from visualizer3d_sans_vbo import Visualizer3D
    FILE = "data/galaxy_100"
    sim = BarnesHutSim(FILE)
    
    # On fixe les limites de vue (ajustez si besoin)
    bounds = ((-5, 5), (-5, 5), (-2, 2))
    visualizer = Visualizer3D(sim.pos, sim.colors, sim.lum, bounds)
    visualizer.camera_distance = 10
    
    def updater(dt_sdl):
        # On utilise le FACTEUR_TEMPS comme dans votre code vectorisé
        # On divise par 10 pour plus de stabilité car votre masse est énorme
        dt_physique = dt_sdl * FACTEUR_TEMPS / 10.0
        return sim.update(dt_physique)

    print("Simulation Barnes-Hut optimisée lancée...")
    visualizer.run_with_updater(updater, dt=0.01)