import pygame
import random

DISPLAY_SCREEN = False

# Initialiser Pygame
pygame.init()

# Définir les dimensions de l'écran
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
if DISPLAY_SCREEN:
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Définir les couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)


class Bird:
    def __init__(self):
        self.x = 50
        self.y = 300
        self.width = 20
        self.height = 20
        self.velocity = 0
        self.gravity = 0.5
        self.jump_strength = -10

    def draw(self):
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height))

    def jump(self):
        self.velocity = self.jump_strength

    def update(self):
        self.velocity += self.gravity
        self.y += self.velocity


class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = 50
        self.gap = 150
        self.height = random.randint(100, 400)
        self.velocity = 5
        self.safe_y = self.height + self.gap - 50

    def draw(self):
        pygame.draw.rect(screen, GREEN, (self.x, 0, self.width, self.height))
        pygame.draw.rect(
            screen,
            GREEN,
            (
                self.x,
                self.height + self.gap,
                self.width,
                SCREEN_HEIGHT - (self.height + self.gap),
            ),
        )

    def update(self):
        self.x -= self.velocity

    def off_screen(self):
        return self.x < -self.width

    def reset(self):
        self.x = SCREEN_WIDTH
        self.height = random.randint(100, 400)
        self.safe_y = self.height + self.gap - 50


class FlappyBirdGameAI:
    def __init__(self):
        self.reset()
        self.clock = pygame.time.Clock()


    def check_floor_or_ceiling_collision(self):
        if self.bird.y > SCREEN_HEIGHT or self.bird.y < 0:
            return True
        return False

    def check_pipe_collision(self):
        isXColliding = (
            self.bird.x + self.bird.width > self.pipe.x
            and self.bird.x < self.pipe.x + self.pipe.width
        )
        isPassingThrough = (
            self.bird.y > self.pipe.height
            and self.bird.y + self.bird.height < self.pipe.height + self.pipe.gap
        )

        if isXColliding and not isPassingThrough:
            return True

        return False

    def draw_score(self):
        font = pygame.font.SysFont(None, 36)
        text = font.render(f"Score: {self.score}", True, BLACK)
        screen.blit(text, (10, 10))

    def reset(self):
        self.bird = Bird()
        self.pipe = Pipe(SCREEN_WIDTH)
        self.score = 0
        self.game_over = False
        self.clock = pygame.time.Clock()

    def get_state(self):
        # Bird's vertical position (y)
        bird_y = self.bird.y
        # Bird's velocity
        bird_velocity = self.bird.velocity
        # Horizontal distance to the next pipe
        distance_to_pipe = self.pipe.x - self.bird.x
        # Vertical distance to the top of the next pipe's gap
        distance_to_top_gap = self.pipe.height - self.bird.y
        # Vertical distance to the bottom of the next pipe's gap
        distance_to_bottom_gap = (self.pipe.height + self.pipe.gap) - self.bird.y

        return [bird_y, bird_velocity, distance_to_pipe, distance_to_top_gap, distance_to_bottom_gap]

    def play_step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Appliquer l'action
        if action == 1:
            self.bird.jump()

        # Mettre à jour l'oiseau et le tuyau
        self.bird.update()
        self.pipe.update()

        reward = 0.1  # Small reward for each step taken

        distance_to_pipe_safe_y = abs(self.bird.y - self.pipe.safe_y)
        reward -= distance_to_pipe_safe_y / SCREEN_HEIGHT * 10 # Negative reward for distance from pipe center



        # Réinitialiser le tuyau et augmenter le score
        if self.pipe.off_screen():
            self.pipe.reset()
            self.score += 1
            reward += 60  # Significant reward for passing through a pipe

        # Vérifier les collisions
        if self.check_pipe_collision():
            reward -= 15
            self.game_over = True

        if self.check_floor_or_ceiling_collision():
            reward -= 30
            self.game_over = True

        # Dessiner tout
        if DISPLAY_SCREEN:
            screen.fill(WHITE)

            self.bird.draw()
            self.pipe.draw()
            pygame.draw.rect(screen, (255,0,0), (0, self.pipe.safe_y, SCREEN_WIDTH, 10))
            self.draw_score()

            pygame.display.flip()

        self.clock.tick(300)

        return reward, self.game_over, self.score
