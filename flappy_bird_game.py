import pygame
import random
import numpy as np

class Bird:
    def __init__(self):
        self.x = 50
        self.y = 300
        self.width = 20
        self.height = 20
        self.velocity = 0
        self.gravity = 0.5
        self.jump_strength = -10

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 0, 0), (self.x, self.y, self.width, self.height))

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

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 255, 0), (self.x, 0, self.width, self.height))
        pygame.draw.rect(screen, (0, 255, 0), (self.x, self.height + self.gap, self.width, 600 - (self.height + self.gap)))

    def update(self):
        self.x -= self.velocity

    def off_screen(self):
        return self.x < -self.width

    def reset(self):
        self.x = 400
        self.height = random.randint(100, 400)
        self.safe_y = self.height + self.gap - 50

class FlappyBirdGameAI:
    def __init__(self, display_screen=False):
        self.display_screen = display_screen
        if self.display_screen:
            pygame.init()
            self.screen = pygame.display.set_mode((400, 600))
            self.clock = pygame.time.Clock()
            self.clock_speed = 40
        self.reset()

    def increase_clock_speed(self):
        self.clock_speed += 20
    
    def decrease_clock_speed(self):
        self.clock_speed = max(20, self.clock_speed - 20)

    def check_floor_or_ceiling_collision(self):
        return self.bird.y > 600 or self.bird.y < 0

    def check_pipe_collision(self):
        is_x_colliding = self.bird.x + self.bird.width > self.pipe.x and self.bird.x < self.pipe.x + self.pipe.width
        is_passing_through = self.bird.y > self.pipe.height and self.bird.y + self.bird.height < self.pipe.height + self.pipe.gap
        return is_x_colliding and not is_passing_through

    def bird_just_passed_pipe(self):
        return self.bird.x == self.pipe.x + self.pipe.width

    def draw_score(self):
        if self.display_screen:
            font = pygame.font.SysFont(None, 36)
            text = font.render(f"Score: {self.score}", True, (0, 0, 0))
            self.screen.blit(text, (10, 10))

    def reset(self):
        self.bird = Bird()
        self.pipe = Pipe(400)
        self.score = 0
        self.game_over = False
        return np.array(self.get_state(), dtype=np.float32)

    def step(self, action):
        reward, game_over, score = self.play_step(action)
        next_state = self.get_state()
        return np.array(next_state, dtype=np.float32), reward, game_over, score

    def get_state(self):
        bird_y = self.bird.y
        # bird_velocity = self.bird.velocity
        # distance_to_pipe = self.pipe.x - self.bird.x - self.bird.width
        # distance_to_top_gap = self.pipe.height - self.bird.y
        # distance_to_bottom_gap = (self.pipe.height + self.pipe.gap) - self.bird.y - self.bird.height
        # return [bird_y, bird_velocity, distance_to_pipe, distance_to_top_gap, distance_to_bottom_gap]

        top_pipe_y = self.pipe.height
        bottom_pipe_y = self.pipe.height + self.pipe.gap
        return [bird_y, top_pipe_y, bottom_pipe_y]

    def pause(self):
        paused = True
        while paused:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    paused = False

    def draw_clock_speed(self):
        if self.display_screen:
            font = pygame.font.SysFont(None, 36)
            text = font.render(f"Clock speed: {self.clock_speed}", True, (0, 0, 0))
            self.screen.blit(text, (10, 50))

    def play_step(self, action):
        if self.display_screen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.pause()
                    if event.key == pygame.K_i:
                        self.increase_clock_speed()
                    if event.key == pygame.K_d:
                        self.decrease_clock_speed()

        if action == 1:
            self.bird.jump()

        self.bird.update()
        self.pipe.update()

        
        # distance_to_pipe_safe_y = abs(self.bird.y - self.pipe.safe_y)
        # reward -= distance_to_pipe_safe_y / 600 * 10

        if self.pipe.off_screen():
            self.pipe.reset()

        if self.bird_just_passed_pipe():
            self.score += 1
            # reward += 120

        if self.check_pipe_collision():
            # reward -= 15
            self.game_over = True

        reward = 0
        if self.check_floor_or_ceiling_collision():
            # reward -= 30
            self.game_over = True
        else :
            reward = 1

        if self.display_screen:
            self.screen.fill((255, 255, 255))
            self.bird.draw(self.screen)
            self.pipe.draw(self.screen)
            pygame.draw.line(self.screen, (255, 0, 0), (0, self.pipe.safe_y), (400, self.pipe.safe_y))
            self.draw_score()
            self.draw_clock_speed()
            pygame.draw.line(self.screen, (255, 0, 0), (self.bird.x + self.bird.width, self.bird.y), (self.pipe.x, self.bird.y))
            pygame.draw.line(self.screen, (255, 0, 0), (self.bird.x + self.bird.width, self.bird.y), (self.pipe.x, self.pipe.height))
            pygame.draw.line(self.screen, (255, 0, 0), (self.bird.x + self.bird.width, self.bird.y + self.bird.height), (self.pipe.x, self.pipe.height + self.pipe.gap))
            pygame.display.flip()
            self.clock.tick(self.clock_speed)

        return reward, self.game_over, self.score

