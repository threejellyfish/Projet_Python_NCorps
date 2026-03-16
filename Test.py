import numpy as np

grid = np.meshgrid(np.linspace(-3,3,20), np.linspace(-3,3,20))

print(grid)

def belongs_to(pos, grid):

    x_coor = pos[0]
    y_coor = pos[1]

    for i in range(0,20) :
        inf = -3 + i*( 3 + 3)/6
        sup = -3 + (i+1)*( 3 + 3)/6
        if x_coor >= inf and x_coor <= sup :
            ix = i
        if y_coor >= inf and y_coor <= sup :
            iy = i
    grid_index = (ix, iy)
    return grid_index

nstars = 100
Grid_pos = np.zeros((nstars,2), dtype=tuple)
for i in range(nstars) :
    Grid_pos[i] = belongs_to(star.pos, grid)

pos = (-2.9, -2)


print("belongs_to :", belongs_to(pos,grid))



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


Box_coord = [ [] for i in range(20*20)]

size = 6/20
for star in points :
    ix = (star[0] - (-3))//size
    iy = (star[1] - (-3))//size

    box_id = 20*ix + iy 
    Box_coord[box_id].append(i)

def update_stars(self):
    for star in system :
        ix = (star[0] - (-3))//size
        iy = (star[1] - (-3))//size

        box_id = 20*ix + iy 

        




