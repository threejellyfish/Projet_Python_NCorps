import os
import numpy as np

G = 1.560339e-13  # Constante gravitationnelle en années-lumière³/(masse_solaire·année²)

class Corps:
    def __init__(self, mass=15, color=np.zeros(3), position=np.zeros(3), speed=np.zeros(3)):
        self.mass = float(mass)  # Conversion en float
        self.color = np.array(color, dtype=np.float32)  # RGB 0-255
        self.position = np.array(position, dtype=np.float64)
        self.speed = np.array(speed, dtype=np.float64)
        self._determine_color()  # Déterminer couleur basée sur masse
        self.grid_pos = self.belongs_to(self.position)

    def _determine_color(self):
        """Détermine la couleur basée sur la masse"""
        if self.mass > 5:
            self.color = np.array([150, 180, 255], dtype=np.float32)  # Bleu-blanc
        elif self.mass > 2:
            self.color = np.array([255, 255, 255], dtype=np.float32)  # Blanc
        elif self.mass > 1:
            self.color = np.array([255, 255, 200], dtype=np.float32)  # Jaune
        else:
            self.color = np.array([250, 150, 100], dtype=np.float32)  # Rouge

    def update(self, acceleration, dt):
        """Met à jour position et vitesse avec l'accélération"""
        self.position += self.speed * dt + 0.5 * acceleration * dt**2
        self.speed += acceleration * dt
        self.grid_pos = self.belongs_to(self.position)

    def update_position(self, position):
        self.position = position

    def update_speed(self, speed):
        self.speed = speed

    def distance_to(self, other):
        """Calcule la distance à un autre corps"""
        return np.linalg.norm(self.position - other.position)

    def print_info(self):
        print(f"Masse: {self.mass:.2f}")
        print(f"Vitesse: {self.speed}")
        print(f"Position: {self.position}")
        print(f"Couleur: {self.color}")

    def belongs_to(self, pos):
        # Clamp pour éviter NameError si l'étoile sort du domaine [-3, 3]
        x_coor = np.clip(pos[0], -3, 3)
        y_coor = np.clip(pos[1], -3, 3)

        ix, iy = 0, 0

        for i in range(0, 20):
            inf = -3 + i * (3 + 3) / 19
            sup = -3 + (i + 1) * (3 + 3) / 19
            if x_coor >= inf and x_coor <= sup:
                ix = i
            if y_coor >= inf and y_coor <= sup:
                iy = i

        grid_index = np.zeros((2))
        grid_index[0] = ix
        grid_index[1] = iy
        return grid_index


class NCorps:
    def __init__(self, n=100, filename=None, bh_mass=1e6, use_grid = False):
        """
        Initialise une galaxie de N étoiles autour d'un trou noir central.

        Paramètres
        ----------
        n        : nombre d'étoiles (hors trou noir)
        filename : chemin vers un fichier de données (optionnel).
                   La première ligne est le trou noir, les suivantes sont des étoiles.
        bh_mass  : masse du trou noir si aucun fichier n'est fourni (défaut 1e6 M☉)
        """
        collection = self._load_from_file(n, filename) if filename else self._generate_random(n, bh_mass)

        self.collection = collection
        self.len = len(collection)
        self.grid = np.array([c.grid_pos for c in collection], dtype=np.int8).reshape(-1, 2)
        self.gravity_centers = np.zeros((20, 20, 3), dtype=np.float32)
        self.use_grid = use_grid

    # ------------------------------------------------------------------
    # Méthodes de construction internes
    # ------------------------------------------------------------------

    @staticmethod
    def _load_from_file(n, filename):
        """Charge trou noir + n étoiles depuis un fichier de données."""
        collection = []

        if not os.path.exists(filename):
            print(f"Fichier {filename} non trouvé, génération aléatoire...")
            return NCorps._generate_random(n, bh_mass=1e6)

        with open(filename, 'r') as f:
            lignes = f.readlines()

        # Première ligne → trou noir
        if len(lignes) > 0:
            data = lignes[0].split()
            if len(data) >= 7:
                bh = Corps(
                    mass=float(data[0]),
                    position=np.array([float(x) for x in data[1:4]]),
                    speed=np.array([float(x) for x in data[4:7]])
                )
                collection.append(bh)

        # Lignes suivantes → étoiles
        for i in range(1, min(n + 1, len(lignes))):
            data = lignes[i].split()
            if len(data) >= 7:
                star = Corps(
                    mass=float(data[0]),
                    position=np.array([float(x) for x in data[1:4]]),
                    speed=np.array([float(x) for x in data[4:7]])
                )
                collection.append(star)

        return collection

    #@staticmethod
    def _generate_random(n, bh_mass=1e6):
        """Génère un trou noir central + n étoiles sur des orbites circulaires."""
        collection = []

        black_hole = Corps(mass=bh_mass, position=np.zeros(3), speed=np.zeros(3))
        collection.append(black_hole)

        for _ in range(n):
            r     = np.random.uniform(1, 3)
            theta = np.random.uniform(0, 2 * np.pi)
            phi   = np.random.uniform(-0.1, 0.1)

            pos = np.array([
                r * np.cos(theta) * np.cos(phi),
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(phi)
            ])

            # Vitesse pour orbite circulaire approximative
            v = np.sqrt(G * bh_mass / r)
            speed = np.array([
                -v * np.sin(theta),
                v * np.cos(theta),
                0.0
            ])

            mass = np.random.uniform(0.5, 10)
            collection.append(Corps(mass=mass, position=pos, speed=speed))

        return collection

    # ------------------------------------------------------------------
    # Calcul des accélérations
    # ------------------------------------------------------------------

    def calculate_accelerations(self):
        """Calcule l'accélération pour chaque corps (force brute O(n²))."""
        n = self.len
        accelerations = np.zeros((n, 3), dtype=np.float64)

        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec    = self.collection[j].position - self.collection[i].position
                    distance = np.linalg.norm(r_vec)
                    if distance > 0:
                        acceleration_magnitude = G * self.collection[j].mass / distance**2
                        accelerations[i] += acceleration_magnitude * (r_vec / distance)

        return accelerations

    def calculate_accelerations2(self):
        """Calcule l'accélération avec approximation par grille (Barnes-Hut simplifié)."""
        accelerations = np.zeros((self.len, 3), dtype=np.float64)
        self.gravity_centers = self.update_gravity_centers()

        for i in range(self.len):
            for j in range(self.len):
                if i != j:
                    center_j = self.gravity_centers[int(self.grid[j, 0]), int(self.grid[j, 1])]

                    if 1/2 * np.linalg.norm(self.collection[i].position - center_j) > 0.3:
                        # Loin : pointer de i vers le centre de masse de la cellule de j
                        r_vec     = center_j - self.collection[i].position
                        cell_mass = self.compute_cell_mass(int(self.grid[j, 0]), int(self.grid[j, 1]))
                    else:
                        # Proche : interaction exacte
                        r_vec     = self.collection[j].position - self.collection[i].position
                        cell_mass = self.collection[j].mass

                    distance = np.linalg.norm(r_vec)
                    if distance > 0:
                        acceleration_magnitude = G * cell_mass / distance**2
                        accelerations[i] += acceleration_magnitude * (r_vec / distance)

        return accelerations

    # ------------------------------------------------------------------
    # Intégrateurs
    # ------------------------------------------------------------------

    def update(self, dt):
        """Euler semi-implicite : met à jour tous les corps."""
        accelerations = self.calculate_accelerations2() if self.use_grid else self.calculate_accelerations()

        for i, corps in enumerate(self.collection):
            corps.update(accelerations[i], dt)
            if self.use_grid:
                self.grid[i] = self.collection[i].grid_pos

    def update2(self, dt):
        """Intégrateur de Verlet."""
        a = self.calculate_accelerations2()
        for i in range(self.len):
            position = self.collection[i].position + self.collection[i].speed * dt + 0.5 * a[i] * dt**2
            self.collection[i].update_position(position)
        a_new = self.calculate_accelerations2()
        for i in range(self.len):
            speed = self.collection[i].speed + 0.5 * (a[i] + a_new[i]) * dt
            self.collection[i].update_speed(speed)

    # ------------------------------------------------------------------
    # Grille / centres de gravité
    # ------------------------------------------------------------------

    def update_gravity_centers(self):
        for i in range(20):
            for j in range(20):
                self.gravity_centers[i, j] = self.compute_gravity_center(i, j)
        return self.gravity_centers

    def compute_gravity_center(self, coord_x, coord_y):
        loc_pos    = []
        loc_masses = []
        for i in range(self.len):
            if self.grid[i, 0] == coord_x and self.grid[i, 1] == coord_y:
                loc_pos.append(self.collection[i].position)
                loc_masses.append(self.collection[i].mass)
        loc_pos    = np.array(loc_pos)
        loc_masses = np.array(loc_masses)
        return np.average(loc_pos, axis=0, weights=loc_masses) if len(loc_pos) > 0 else np.zeros(3)

    def compute_cell_mass(self, coord_x, coord_y):
        """Masse totale des corps dans une cellule de la grille."""
        return sum(
            self.collection[i].mass
            for i in range(self.len)
            if self.grid[i, 0] == coord_x and self.grid[i, 1] == coord_y
        )

    # ------------------------------------------------------------------
    # Accesseurs pour la visualisation
    # ------------------------------------------------------------------

    def get_points(self):
        points = np.zeros((self.len, 3), dtype=np.float32)
        for i, corps in enumerate(self.collection):
            points[i] = corps.position.astype(np.float32)
        return points

    def get_colors(self):
        colors = np.zeros((self.len, 3), dtype=np.float32)
        for i, corps in enumerate(self.collection):
            colors[i] = corps.color
        return colors

    def get_masses(self):
        masses = np.zeros(self.len, dtype=np.float32)
        for i, corps in enumerate(self.collection):
            masses[i] = corps.mass
        return masses


def main():
    # --- Initialisation directe : NCorps construit la galaxie en interne ---
    galaxy = NCorps(n=100, filename="data/galaxy_1000",use_grid = True)

    # Préparation des données pour la visualisation
    points      = galaxy.get_points()
    colors      = galaxy.get_colors()
    masses      = galaxy.get_masses()
    luminosities = np.clip(masses / masses.max(), 0.3, 1.0).astype(np.float32)

    x_min, x_max = points[:, 0].min() - 0.5, points[:, 0].max() + 0.5
    y_min, y_max = points[:, 1].min() - 0.5, points[:, 1].max() + 0.5
    z_min, z_max = points[:, 2].min() - 0.5, points[:, 2].max() + 0.5
    bounds = ((x_min, x_max), (y_min, y_max), (z_min, z_max))

    print(f"Nombre de corps: {galaxy.len}")
    print(f"Limites: {bounds}")

    def updater(dt):
        galaxy.update(dt)
        return galaxy.get_points(), galaxy.get_colors(), luminosities

    try:
        from visualizer3d_sans_vbo import Visualizer3D
        visualizer = Visualizer3D(
            points=points,
            colors=colors,
            luminosities=luminosities,
            bounds=bounds
        )
        visualizer.run_with_updater(updater, dt=0.01)
    except Exception as e:
        print(f"Erreur de visualisation: {e}")
        for step in range(100):
            galaxy.update(0.001)
            if step % 10 == 0:
                print(f"Étape {step}")


if __name__ == "__main__":
    main()
