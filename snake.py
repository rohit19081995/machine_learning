import random
import pygame,sys
from pygame.locals import *
from pygame.time import Clock

DEFAULT_RESOLUTION = [15,15]
PIXELS_PER_TILE = 10
FPS = 120
BACKGROUND_COLOR = (255,255,255)
SNAKE_COLOR = (0,0,0)
HEAD_COLOR = (100,100,255)
FOOD_COLOR = (255, 100, 100)

class Snake(object):

	def __init__(self, resolution=DEFAULT_RESOLUTION, display=False):
		self.resolution = resolution
		self.head = [3,0]
		self.snake = [[2,0], [1,0], [0,0]]
		self.direction = 'right'
		self.food = self.generate_new_food()
		self.score = 0
		self.done = False
		self.display = display
		if display:
			pygame.init()
			surface_size = (self.resolution[0]*PIXELS_PER_TILE, self.resolution[1]*PIXELS_PER_TILE + 30)
			self.display_surface = pygame.display.set_mode(surface_size)
			pygame.display.set_caption('Snake Game?')
			self.clock = pygame.time.Clock()
			pygame.font.init()
			self.font = pygame.font.SysFont('Comic Sans MS', 30)


	def generate_new_food(self):
		food_x = random.randint(0, self.resolution[0]-1)
		food_y = random.randint(0, self.resolution[1]-1)
		while [food_x, food_y] in self.snake or [food_x, food_y] is self.head:
			food_x = random.randint(0, self.resolution[0]-1)
			food_y = random.randint(0, self.resolution[1]-1)
		return [food_x, food_y]

	def next(self, keypress=[1,0,0,0,0]):
		# Nothing pressed
		if keypress[1] == 1 and self.direction != 'right':
			self.direction = 'left'
		if keypress[2] == 1 and self.direction != 'left':
			self.direction = 'right'
		if keypress[3] == 1 and self.direction != 'down':
			self.direction = 'up'
		if keypress[4] == 1 and self.direction != 'up':
			self.direction = 'down'

		# make snake head the body
		self.snake.insert(0, list(self.head))
		# Acctually move the snake
		if self.direction == 'left':
			if self.head[0] == 0:
				self.head[0] = self.resolution[0]-1
			else:
				self.head[0] -= 1
		if self.direction == 'right':
			if self.head[0] == self.resolution[0]-1:
				self.head[0] = 0
			else:
				self.head[0] += 1
		if self.direction == 'up':
			if self.head[1] == 0:
				self.head[1] = self.resolution[1]-1
			else:
				self.head[1] -= 1
		if self.direction == 'down':
			if self.head[1] == self.resolution[1]-1:
				self.head[1] = 0
			else:
				self.head[1] += 1

		# if snake is on food
		if self.head == self.food:
			self.food = self.generate_new_food()
			self.score += 1
		# if snake eats itself
			self.done = True
		# if snake is not on food
		else:
			self.snake.pop()

		if self.display:
			for event in pygame.event.get():
				if event.type == QUIT:
					pygame.quit()
					sys.exit()
			self.display_surface.fill(BACKGROUND_COLOR)
			for tile in self.snake:
				self.blit_tile(tile, SNAKE_COLOR)
			self.blit_tile(self.head, HEAD_COLOR)
			self.blit_tile(self.food, FOOD_COLOR)
			ts = self.font.render('%d'%self.score, False, (0,0,0))
			self.display_surface.blit(ts, (10, PIXELS_PER_TILE*self.resolution[1] + 10))
			pygame.display.update()
			self.clock.tick(FPS)


	def blit_tile(self, tile, color):
		x = tile[0]*PIXELS_PER_TILE+1
		y = tile[1]*PIXELS_PER_TILE+1
		pygame.draw.rect(self.display_surface, color, (x,y,PIXELS_PER_TILE-1, PIXELS_PER_TILE-1))
