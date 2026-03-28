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

        x_coor = pos[0]
        y_coor = pos[1]

        for i in range(0,20) :
            inf = -3 + i*( 3 + 3)/19
            sup = -3 + (i+1)*( 3 + 3)/19
            if x_coor >= inf and x_coor <= sup :
                ix = i
            if y_coor >= inf and y_coor <= sup :
                iy = i
        
        #print("x_coor :", x_coor)
        #print("y_coor :", y_coor)
        grid_index = np.zeros((2))
        grid_index[0] = ix 
        grid_index[1] = iy
        return grid_index

    #nstars = 100
    #Grid_pos = np.zeros((nstars,2), dtype=tuple)
    #for i in range(nstars) :
    #    Grid_pos[i] = belongs_to(star.pos, grid)



class NCorps: 
    def __init__(self, collection=None):
        self.collection = collection if collection is not None else []
        self.len = len(self.collection)
        self.grid = np.zeros((self.len, 2), dtype=np.int8)
        self.gravity_centers = np.zeros((20,20,3), dtype=np.float32)
    
    def add(self, corps): 
        self.collection.append(corps)
        self.len += 1 
        self.grid = np.concatenate((self.grid, np.expand_dims(corps.grid_pos, axis = 0)))
        #self.gravity_center = np.concatenate((self.grid, np.expand_dims(self.compute_gravity_center(self.grid[self.len -1, 0], self.grid[self.len -1, 1]), axis = 0)))
        #print("self.grid :", self.grid)
    
    def calculate_accelerations(self):
        """Calcule l'accélération pour chaque corps due à l'attraction gravitationnelle"""
        n = self.len
        accelerations = np.zeros((n, 3), dtype=np.float64)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Vecteur de i à j
                    r_vec = self.collection[j].position - self.collection[i].position
                    distance = np.linalg.norm(r_vec)
                    
                    if distance > 0:  # Éviter division par zéro
                        # Force gravitationnelle: F = G * m_i * m_j / r²
                        # Accélération: a = F / m_i = G * m_j / r² * (r_vec / r)
                        acceleration_magnitude = G * self.collection[j].mass / (distance**2)
                        acceleration_direction = r_vec / distance
                        accelerations[i] += acceleration_magnitude * acceleration_direction
        
        return accelerations
    
    def calculate_accelerations2(self):
        "Grid"
        accelerations = np.zeros((self.len, 3), dtype=np.float64)
        #self.gravity_centers = np.ones((20,20,3), dtype=np.int16)
        self.gravity_centers = self.update_gravity_centers()
        #print("self.grid :", self.grid)
        #print("self.gravity_centers :", self.gravity_centers[6:11, 6:12,:])
        for i in range(self.len):
            #print(" star id :", i)
            for j in range(self.len):
                if i != j:
                    #print("self.grid[j,0] :", self.grid[j,0])
                    #print("self.grid[j,1] :", self.grid[j,1])
                    #print("j", j)
                    center_j = self.gravity_centers[int(self.grid[j,0]), int(self.grid[j,1])]
                    if 1/2* np.linalg.norm(self.collection[i].position - center_j) > 0.3 : #diametre = 0.3
                        #print("center_ij :", center_ij)
                        r_vec = self.collection[i].position - center_j
                        #print("r_vec :", r_vec)
                    else :
                        r_vec = self.collection[j].position - self.collection[i].position
                    
                    distance = np.linalg.norm(r_vec)
                    #print("distance :", distance)
                    
                    if distance > 0:  # Éviter division par zéro
                        # Force gravitationnelle: F = G * m_i * m_j / r²
                        # Accélération: a = F / m_i = G * m_j / r² * (r_vec / r)
                        acceleration_magnitude = G * self.collection[j].mass / (distance**2)
                        acceleration_direction = r_vec / distance
                        accelerations[i] += acceleration_magnitude * acceleration_direction
        
        return accelerations 

    
    def update(self, dt):
        """Met à jour tous les corps avec leurs accélérations"""
        accelerations = self.calculate_accelerations2()
        #print("accelerations 1 :", accelerations)
        
        for i, corps in enumerate(self.collection):
            #print(" star id :", i)
            corps.update(accelerations[i], dt)
            self.grid[i] = self.collection[i].grid_pos

        self.gravity_centers = self.update_gravity_centers()

            
    def update2(self, dt):
        "Verlet"
        #print("Verlet accelerations :", acceleration)
        a = self.calculate_accelerations2()
        for i in range(self.len):
            position = self.collection[i].position + self.collection[i].speed*dt + 0.5* a*dt*dt
            self.collection[i].update_position(position)
        a_new = self.calculate_accelerations2()
        for i in range(self.len):

            speed = self.collection[i].speed + 0.5*(a + a_new)*dt
            self.collecion[i].update_speed(speed)
        

    def update_gravity_centers(self):
        for i in range(20):
            for j in range(20):
                self.gravity_centers[i,j] = self.compute_gravity_center(i,j)
                #print("self.compute_gravity_center(i,j) :",i,j,self.compute_gravity_center(i,j), self.gravity_centers[i,j])
        #print("self.gravity_centers :", self.gravity_centers[5:13, 5:13,:])
        return self.gravity_centers

    def compute_gravity_center(self,coord_x,coord_y): 
        #print("coord_x :", coord_x)
        #print("coord_y :", coord_y)
        loc_pos = []
        for i in range(self.len):
            #print("self.grid[i,0] :", self.grid[i,0])
            #print("self.grid[i,1] :", self.grid[i,1])
            #print("self.grid[i,0] == coord_x :", self.grid[i,0] == coord_x )
            #print("self.grid[i,1] == coord_y :", self.grid[i,1] == coord_y )

            if self.grid[i,0] == coord_x and self.grid[i,1] == coord_y :
                loc_pos.append(self.collection[i].position)
        loc_pos = np.array(loc_pos)
        #print("loc_pos :",loc_pos)
        #print("np.mean(loc_pos) :", np.mean(loc_pos,axis =0))
        return np.mean(loc_pos, axis=0) if np.shape(loc_pos)[0] > 0 else np.zeros((3,))
    
    
    def get_points(self):
        """Retourne les positions pour la visualisation"""
        points = np.zeros((self.len, 3), dtype=np.float32)
        for i, corps in enumerate(self.collection):
            points[i] = corps.position.astype(np.float32)
        return points
    
    def get_colors(self):
        """Retourne les couleurs pour la visualisation (0-255)"""
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
    
    # Paramètres
    nstars = 100
    filename = "data/galaxy_1000"
    
    # Création de la galaxie
    galaxy = NCorps()


    
    # Lecture du fichier de données
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lignes = f.readlines()
        
        # Création du trou noir
        if len(lignes) > 0:
            data = lignes[0].split()
            if len(data) >= 7:
                bh_mass = float(data[0])
                bh_pos = np.array([float(x) for x in data[1:4]])
                bh_speed = np.array([float(x) for x in data[4:7]])

                black_hole = Corps(mass=bh_mass, position=bh_pos, speed=bh_speed)

                black_hole._determine_color()
                galaxy.add(black_hole)
        
        # Création des étoiles
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
        # Génération aléatoire si fichier inexistant
        print(f"Fichier {filename} non trouvé, génération aléatoire...")
        
        # Trou noir central
        black_hole = Corps(mass=1e6, position=np.zeros(3))
        galaxy.add(black_hole)
        
        # Étoiles en orbite
        for i in range(nstars):
            # Position aléatoire dans un disque
            r = np.random.uniform(1, 3)
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(-0.1, 0.1)  # Léger décalage vertical
            
            pos = np.array([
                r * np.cos(theta) * np.cos(phi),
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(phi)
            ])
            
            # Vitesse pour orbite circulaire approximative
            v = np.sqrt(G * black_hole.mass / r)
            speed = np.array([
                -v * np.sin(theta),
                v * np.cos(theta),
                0
            ])
            
            mass = np.random.uniform(0.5, 10)
            star = Corps(mass=mass, position=pos, speed=speed)
            galaxy.add(star)
    
    # Préparation des données pour la visualisation
    points = galaxy.get_points()

    max_pos = np.zeros((3))
    max_norm = 0.0
    for p in points :
        if np.linalg.norm(p) > max_norm :
            max_norm = np.linalg.norm(p)
            max_pos = p

    max_x = np.argmax(points)

    bounds = ((-max_pos[0], max_pos[0]))

    grid = np.linspace(-3,3, 1000)

    colors = galaxy.get_colors()
    masses = galaxy.get_masses()
    
    # Luminosités basées sur la masse
    luminosities = np.clip(masses / masses.max(), 0.3, 1.0).astype(np.float32)
    
    # Calcul des limites de l'espace
    all_points = points
    x_min, x_max = all_points[:, 0].min() - 0.5, all_points[:, 0].max() + 0.5
    y_min, y_max = all_points[:, 1].min() - 0.5, all_points[:, 1].max() + 0.5
    z_min, z_max = all_points[:, 2].min() - 0.5, all_points[:, 2].max() + 0.5
    bounds = ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    
    print(f"Nombre de corps: {galaxy.len}")
    print(f"Limites: {bounds}")
    
    # Fonction de mise à jour pour la visualisation
    def updater(dt):
        galaxy.update(dt)
        return galaxy.get_points(), galaxy.get_colors(), luminosities
    
    # Import et lancement de la visualisation
    try:
        from visualizer3d_sans_vbo import Visualizer3D
        
        # Correction: passer les bons paramètres dans le bon ordre
        visualizer = Visualizer3D(
            points=points,  # Premier paramètre: points
            colors=colors,  # Deuxième paramètre: couleurs (0-255)
            luminosities=luminosities,  # Troisième paramètre: luminosités (0-1)
            bounds=bounds  # Quatrième paramètre: limites
        )
        
        # Adaptation de la méthode run pour accepter notre updater
        visualizer.run_with_updater(updater, dt=0.01)
        
    except Exception as e:
        print(f"Erreur de visualisation: {e}")
        # Simulation sans visualisation
        for step in range(100):
            galaxy.update(0.001)
            if step % 10 == 0:
                print(f"Étape {step}")


if __name__ == "__main__":
    main()