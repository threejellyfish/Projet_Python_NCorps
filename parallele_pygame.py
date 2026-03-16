parallele_pygame.py << 'EOF'
import pygame
import threading
import time
import random
import math
from queue import Queue

class CalculateurParallele:
    def __init__(self):
        self.resultats = Queue()
        self.taches = Queue()
        self.threads = []
        self.running = True
        
    def calcul_complexe(self, x, y):
        """Simule un calcul mathématique intensif"""
        time.sleep(0.1)  # Simulation de calcul
        return math.sin(x) * math.cos(y) * 100
    
    def worker(self, id_thread):
        """Fonction exécutée par chaque thread"""
        while self.running:
            try:
                x, y = self.taches.get(timeout=0.1)
                resultat = self.calcul_complexe(x, y)
                self.resultats.put((x, y, resultat, id_thread))
            except:
                pass
    
    def demarrer_threads(self, nb_threads=4):
        for i in range(nb_threads):
            t = threading.Thread(target=self.worker, args=(i,))
            t.daemon = True
            t.start()
            self.threads.append(t)
    
    def arreter(self):
        self.running = False

# Initialisation Pygame
pygame.init()
screen = pygame.display.set_mode((1000, 700))
pygame.display.set_caption("Calcul Parallèle avec Pygame")
font = pygame.font.Font(None, 24)
clock = pygame.time.Clock()

# Couleurs
NOIR = (0, 0, 0)
BLANC = (255, 255, 255)
COULEURS_THREADS = [
    (255, 100, 100),  # Rouge
    (100, 255, 100),  # Vert
    (100, 100, 255),  # Bleu
    (255, 255, 100),  # Jaune
]

# Initialisation du calculateur
calc = CalculateurParallele()
calc.demarrer_threads(4)

# Générer des points de calcul
points = [(random.uniform(-3, 3), random.uniform(-3, 3)) for _ in range(50)]
for x, y in points:
    calc.taches.put((x, y))

resultats_visuels = []
running = True
iteration = 0

while running and iteration < 300:  # 5 secondes à 60 FPS
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Récupérer les résultats disponibles
    while not calc.resultats.empty():
        x, y, valeur, thread_id = calc.resultats.get()
        resultats_visuels.append({
            'x': x,
            'y': y,
            'valeur': valeur,
            'thread': thread_id,
            'age': 100
        })
    
    # Ajouter de nouveaux calculs
    if calc.taches.qsize() < 10 and iteration % 30 == 0:
        for _ in range(5):
            calc.taches.put((
                random.uniform(-3, 3),
                random.uniform(-3, 3)
            ))
    
    # Dessin
    screen.fill(NOIR)
    
    # Titre
    title = font.render(f"Calculs parallèles - Threads actifs: 4", True, BLANC)
    screen.blit(title, (10, 10))
    
    # Stats
    stats = font.render(f"Résultats: {len(resultats_visuels)} | En attente: {calc.taches.qsize()}", True, BLANC)
    screen.blit(stats, (10, 40))
    
    # Visualisation des résultats
    center_x, center_y = 500, 350
    for r in resultats_visuels[:]:
        # Conversion des coordonnées
        screen_x = int(center_x + r['x'] * 100)
        screen_y = int(center_y - r['y'] * 100)
        
        if 0 <= screen_x < 1000 and 0 <= screen_y < 700:
            # Couleur basée sur le thread et la valeur
            couleur_base = COULEURS_THREADS[r['thread']]
            intensite = min(255, int(abs(r['valeur']) * 2))
            couleur = tuple(min(255, c * intensite // 100) for c in couleur_base)
            
            pygame.draw.circle(screen, couleur, (screen_x, screen_y), 5)
            
            # Afficher l'ID du thread
            id_text = font.render(str(r['thread']), True, BLANC)
            screen.blit(id_text, (screen_x - 5, screen_y - 15))
        
        r['age'] -= 1
        if r['age'] <= 0:
            resultats_visuels.remove(r)
    
    pygame.display.flip()
    clock.tick(60)
    iteration += 1

calc.arreter()
pygame.quit()
EOF
