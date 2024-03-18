import neat
import os
import pygame
import numpy as np

# Set the window size
WIN_WIDTH = 600
WIN_HEIGHT = 600

OBSTACLE_GAP = 120
OBSTACLE_RANGE = [WIN_HEIGHT//4, 3*WIN_HEIGHT//4]

generation = 0
best_fitness = 0
best_score = 0

pygame.init()
SCREEN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Read in pipe img
PIPE_IMG = pygame.image.load(os.path.join("imgs", "pipe.png"))
CLOUD_IMG = pygame.image.load(os.path.join("imgs", "cloud.png"))

class MovableObject:
    def __init__(self,img,y,vel = 6):
        self.img = img
        self.x = WIN_WIDTH + img.get_width()
        self.y = y
        self.vel = vel
    
    def draw(self,screen):
        screen.blit(self.img, (self.x, self.y))
    
    def update(self):
        self.x -= self.vel
class Bird:
    def __init__(self,x,y,radius=20,color = (255,0,0)):
        self.x = y
        self.y = y
        self.vel = 0
        self.radius = radius
        self.gravity = 20
        self.color = color
        self.score = 0
    
    def jump(self):
        self.vel = -50
    def update(self):
        dt = 1/3
        self.vel += self.gravity * dt
        self.y += self.vel * dt

    def draw_inputs(self,screen, obstacle):
        pygame.draw.line(screen, (0,0,0), (self.x, self.y), (obstacle.x, obstacle.top), 2)
        pygame.draw.line(screen, (0,0,0), (self.x, self.y), (obstacle.x, obstacle.bottom), 2)

    def draw(self,screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

class Obstacle:
    def __init__(self,speed = 8):
        self.passed = False
        self.width = 50
        self.x = WIN_WIDTH + self.width
        #random between OBSTACLE_RANGE
        self.top = np.random.randint(OBSTACLE_RANGE[0], OBSTACLE_RANGE[1])
        self.bottom = self.top + OBSTACLE_GAP
        self.color = (0,255,0)
        self.speed = speed

    def update(self):
        self.x -= self.speed
    
    def draw(self,screen):
        screen.blit(pygame.transform.flip(PIPE_IMG, False, True), (self.x, self.top - PIPE_IMG.get_height()))
        screen.blit(PIPE_IMG, (self.x, self.bottom))


    def collide(self, bird):
        # collision between circle of bird and the two rectangles of the obstacle (pipe)
        if bird.y - bird.radius < self.top or bird.y + bird.radius > self.bottom:
            if bird.x + bird.radius > self.x and bird.x - bird.radius < self.x + self.width:
                return True
        return False



def main(genomes, config):
    global SCREEN, generation, best_fitness, best_score
    clock = pygame.time.Clock()

    run = True
    generation += 1

    # Populate the required lists
    networks = []
    birds = []
    ge = []
    for _,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        networks.append(net)
        birds.append(Bird(100,100))
        g.fitness = 0
        ge.append(g)
    
    obstacles = [Obstacle()]
    moveable_objects = [MovableObject(CLOUD_IMG, 100, 2)]

    closest_obstacle = obstacles[0] # The pipe that is closest to the bird
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        
        for i,bird in enumerate(birds):
            ge[i].fitness += 0.2 # Survival bonus

            # Keep track of the best fitness achieved across all generations
            if ge[i].fitness > best_fitness:
                best_fitness = ge[i].fitness

            # Calculate the distance between the bird and the closest pipe, top and bottom part
            top_distance = abs(bird.y - closest_obstacle.top)
            bottom_distance = abs(bird.y - closest_obstacle.bottom)

            # Pass the inputs to the neural network and decide whether to jump or not
            output = networks[i].activate((bird.y, top_distance, bottom_distance))
            if output[0] > 0.5:
                bird.jump()
            
            bird.update()

            # Check if the bird is out of bounds
            if bird.y - bird.radius < 0 or bird.y + bird.radius > WIN_HEIGHT:
                birds.pop(i)
                networks.pop(i)
                ge[i].fitness -= 2
                ge.pop(i)
        
        for i,obstacle in enumerate(obstacles):
            obstacle.update()
            # Remove obstacle if it is out of bounds
            if obstacle.x + obstacle.width < 0:
                obstacles.remove(obstacle)

            for j,bird in enumerate(birds):
                if obstacle.collide(bird):
                    ge[j].fitness -= 1
                    birds.pop(j)
                    networks.pop(j)
                    ge.pop(j)

        # If no birds are left end this generation
        if len(birds) == 0:
            break
        elif closest_obstacle.x + closest_obstacle.width < birds[0].x - birds[0].radius:
            # Birds have crossed the pipe, create a new one and increase fitness
            obstacles.append(Obstacle())
            closest_obstacle = obstacles[-1]
            for i,bird in enumerate(birds):
                ge[i].fitness += 5
                bird.score += 1
                if bird.score > best_score:
                    best_score = bird.score
                if ge[i].fitness > best_fitness:
                    best_fitness = ge[i].fitness

        # ---- Cloud logic ----
        for moveable_object in moveable_objects:
            moveable_object.update()
            if moveable_object.x + moveable_object.img.get_width() < 0:
                moveable_objects.remove(moveable_object)
        
        if len(moveable_objects) == 0:
            height = np.random.randint(20,WIN_HEIGHT//4)
            moveable_objects.append(MovableObject(CLOUD_IMG, height, 2))
        elif moveable_objects[-1].x < WIN_WIDTH//2:
            height = np.random.randint(20,WIN_HEIGHT//4)
            moveable_objects.append(MovableObject(CLOUD_IMG, height, 2))
        # ---------------------
        
        # ---- Drawing objects and debug info ----
        SCREEN.fill((135, 206, 235))
        for moveable_object in moveable_objects:
            moveable_object.draw(SCREEN)
        for bird in birds:
            bird.draw(SCREEN)
            bird.draw_inputs(SCREEN,closest_obstacle)
        for obstacle in obstacles:
            obstacle.draw(SCREEN)
        
        font = pygame.font.SysFont("arial", 20)
        text = font.render(f"Generation: {generation}", True, (0,0,0))
        SCREEN.blit(text, (10,10))
        # round to 2 decimal places
        text = font.render(f"Best Fitness: {round(best_fitness,2)}", True, (0,0,0))
        SCREEN.blit(text, (10,30))
        text = font.render(f"Best Score: {best_score}", True, (0,0,0))
        SCREEN.blit(text, (10,50))
        text = font.render(f"Current Score: {birds[0].score}", True, (0,0,0))
        SCREEN.blit(text, (10,70))
        # ----------------------------------------
        pygame.display.update()

def run(config_path):
    pygame.init()
    config = neat.config.Config(
        neat.DefaultGenome, 
        neat.DefaultReproduction, 
        neat.DefaultSpeciesSet, 
        neat.DefaultStagnation, 
        config_path)
    p = neat.Population(config)
    winner = p.run(main, 50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_config.txt")
    run(config_path)

    
