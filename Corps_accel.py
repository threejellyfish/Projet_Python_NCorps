import numpy as np

G = 1.560339e-13  # Constante gravitationnelle en années-lumière³/(masse_solaire·année²)

class Corps: 
    def __init__(self, mass=15, color=np.zeros(3), position=np.zeros(3), speed=np.zeros(3)):
        self.mass = float(mass)
        self.color = np.array(color, dtype=np.float32)
        self.position = np.array(position, dtype=np.float64)
        self.speed = np.array(speed, dtype=np.float64)
        self._determine_color()
        self.grid_pos = self.belongs_to(self.position)
    
    def _determine_color(self):
        """Détermine la couleur basée sur la masse"""
        if self.mass > 5:
            self.color = np.array([150, 180, 255], dtype=np.float32)  # Bleu-blanc
        elif self.mass > 2:
            self.color = np.array([255, 255, 255], dtype=np.float32)  # Blanc
        elif self.mass >= 1:  # CORRECTION : >= au lieu de >
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
        x_coor = pos[0]
        y_coor = pos[1]

        # CORRECTION : initialisation pour éviter UnboundLocalError si pos hors domaine
        ix, iy = 0, 0

        for i in range(20):
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
    def __init__(self, collection=None):
        self.collection = collection if collection is not None else []
        self.len = len(self.collection)
        self.grid = np.zeros((self.len, 2), dtype=np.int8)
        self.gravity_centers = np.zeros((20, 20, 3), dtype=np.float32)
    
    def add(self, corps): 
        self.collection.append(corps)
        self.len += 1
        self.grid = np.concatenate((self.grid, np.expand_dims(corps.grid_pos, axis=0)))
    
    def calculate_accelerations(self):
        """Calcule l'accélération pour chaque corps due à l'attraction gravitationnelle"""
        n = self.len
        accelerations = np.zeros((n, 3), dtype=np.float64)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = self.collection[j].position - self.collection[i].position
                    distance = np.linalg.norm(r_vec)
                    
                    if distance > 0:
                        acceleration_magnitude = G * self.collection[j].mass / (distance**2)
                        acceleration_direction = r_vec / distance
                        accelerations[i] += acceleration_magnitude * acceleration_direction
        
        return accelerations
    
    def calculate_accelerations2(self):
        """Calcul avec grille et centres de gravité"""
        accelerations = np.zeros((self.len, 3), dtype=np.float64)
        self.gravity_centers = self.update_gravity_centers()

        for i in range(self.len):
            for j in range(self.len):
                if i != j:
                    center_j = self.gravity_centers[int(self.grid[j, 0]), int(self.grid[j, 1])]
                    if 0.5 * np.linalg.norm(self.collection[i].position - center_j) > 0.3:
                        r_vec = center_j - self.collection[i].position
                    else:
                        r_vec = self.collection[j].position - self.collection[i].position
                    
                    distance = np.linalg.norm(r_vec)
                    
                    if distance > 0:
                        acceleration_magnitude = G * self.collection[j].mass / (distance**2)
                        acceleration_direction = r_vec / distance
                        accelerations[i] += acceleration_magnitude * acceleration_direction
        
        return accelerations

    def update(self, dt):
        """Met à jour tous les corps avec leurs accélérations"""
        accelerations = self.calculate_accelerations()
        
        for i, corps in enumerate(self.collection):
            corps.update(accelerations[i], dt)
            self.grid[i] = self.collection[i].grid_pos

        self.gravity_centers = self.update_gravity_centers()

    def update2(self, dt):
        """Verlet"""
        a = self.calculate_accelerations2()
        for i in range(self.len):
            position = self.collection[i].position + self.collection[i].speed * dt + 0.5 * a * dt * dt
            self.collection[i].update_position(position)
        a_new = self.calculate_accelerations2()
        for i in range(self.len):
            speed = self.collection[i].speed + 0.5 * (a + a_new) * dt
            self.collection[i].update_speed(speed)  # CORRECTION : "collection" au lieu de "collecion"

    def update_gravity_centers(self):
        for i in range(20):
            for j in range(20):
                self.gravity_centers[i, j] = self.compute_gravity_center(i, j)
        return self.gravity_centers

    def compute_gravity_center(self, coord_x, coord_y):
        """CORRECTION : moyenne pondérée par les masses (centre de gravité réel)"""
        loc_pos = []
        loc_masses = []
        for i in range(self.len):
            if self.grid[i, 0] == coord_x and self.grid[i, 1] == coord_y:
                loc_pos.append(self.collection[i].position)
                loc_masses.append(self.collection[i].mass)
        loc_pos = np.array(loc_pos)
        loc_masses = np.array(loc_masses)
        return np.average(loc_pos, axis=0, weights=loc_masses) if len(loc_pos) > 0 else np.zeros((3,))
    
    def get_points(self):
        """Retourne les positions pour la visualisation"""
        points = np.zeros((self.len, 3), dtype=np.float32)
        for i, corps in enumerate(self.collection):
            points[i] = corps.position.astype(np.float32)
        return points
    
    def get_colors(self):
        """Retourne les couleurs pour la visualisation"""
        colors = np.zeros((self.len, 3), dtype=np.float32)
        for i, corps in enumerate(self.collection):
            colors[i] = corps.color
        return colors
    
    def get_masses(self):
        """Retourne les masses"""
        masses = np.zeros(self.len, dtype=np.float32)
        for i, corps in enumerate(self.collection):
            masses[i] = corps.mass
        return masses


def main():
    import os
    
    nstars = 100
    filename = "data/galaxy_1000"
    
    galaxy = NCorps()
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lignes = f.readlines()
        
        if len(lignes) > 0:
            data = lignes[0].split()
            if len(data) >= 7:
                bh_mass = float(data[0])
                bh_pos = np.array([float(x) for x in data[1:4]])
                bh_speed = np.array([float(x) for x in data[4:7]])
                black_hole = Corps(mass=bh_mass, position=bh_pos, speed=bh_speed)
                black_hole._determine_color()
                galaxy.add(black_hole)
        
        for i in range(1, min(nstars + 1, len(lignes))):
            data = lignes[i].split()
            if len(data) >= 7:
                star_mass = float(data[0])
                star_pos = np.array([float(x) for x in data[1:4]])
                star_speed = np.array([float(x) for x in data[4:7]])
                star = Corps(mass=star_mass, position=star_pos, speed=star_speed)
                star._determine_color()
                galaxy.add(star)
    else:
        print(f"Fichier {filename} non trouvé, génération aléatoire...")
        
        black_hole = Corps(mass=1e6, position=np.zeros(3))
        galaxy.add(black_hole)
        
        for i in range(nstars):
            r = np.random.uniform(1, 3)
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(-0.1, 0.1)
            
            pos = np.array([
                r * np.cos(theta) * np.cos(phi),
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(phi)
            ])
            
            v = np.sqrt(G * black_hole.mass / r)
            speed = np.array([
                -v * np.sin(theta),
                v * np.cos(theta),
                0
            ])
            
            mass = np.random.uniform(0.5, 10)
            star = Corps(mass=mass, position=pos, speed=speed)
            galaxy.add(star)
    
    points = galaxy.get_points()
    colors = galaxy.get_colors()
    masses = galaxy.get_masses()
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