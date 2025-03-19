
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:13:01 2025

@author: Vivi Rodriguez
"""

"""
Entrenamiento de agente logístico con Q-Learning 
"""

import numpy as np
import random
import gym
import pygame
import time
import matplotlib.pyplot as plt
from gym import spaces

class WarehouseEnv(gym.Env):
    def __init__(self, grid_size=(10, 10), energy=100, num_dynamic_robots=2):
        super(WarehouseEnv, self).__init__()
        self.grid_size = grid_size
        self.obstacles = {(3, 3), (3, 4), (3, 5), (6, 6), (7, 6)}
        self.pickup_point = (1, 1)
        self.delivery_point = (8, 8)
        self.robot_position = (0, 0)
        self.item_picked = False
        self.energy = energy
        self.deliveries = 0
        self.collisions = 0  # Contador de colisiones
        self.steps = 0
        self.dynamic_robots = [(random.randint(0, grid_size[0]-1), random.randint(0, grid_size[1]-1)) for _ in range(num_dynamic_robots)]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([grid_size[0]-1, grid_size[1]-1, 1, energy]),
            dtype=np.int32
        )
        self.reward_structure = {
            'movimiento': -1,
            'colisión': -10,
            'entrega_exitosa': 10,
            'agotar_energía': -5
        }

    def reset(self):
        self.robot_position = (0, 0)
        self.item_picked = False
        self.energy = 100
        self.deliveries = 0
        self.collisions = 0
        self.steps = 0
        return np.array([*self.robot_position, int(self.item_picked), self.energy], dtype=np.int32)

    def step(self, action):
        x, y = self.robot_position
        reward = -1
        done = False
        self.steps += 1

        # Movimiento del robot
        new_x, new_y = x, y
        if action == 0 and y > 0:  # Arriba
            new_y -= 1
        elif action == 1 and y < self.grid_size[1] - 1:  # Abajo
            new_y += 1
        elif action == 2 and x > 0:  # Izquierda
            new_x -= 1
        elif action == 3 and x < self.grid_size[0] - 1:  # Derecha
            new_x += 1

        # Verificar colisiones
        if (new_x, new_y) in self.obstacles or (new_x, new_y) in self.dynamic_robots:
            self.collisions += 1
            reward -= 10
        else:
            self.robot_position = (new_x, new_y)

        # Recoger/Entregar automáticamente
        if self.robot_position == self.pickup_point:
            self.item_picked = True
        if self.robot_position == self.delivery_point and self.item_picked:
            reward = 10
            self.item_picked = False
            self.deliveries += 1

        # Mover robots dinámicos
        self.move_dynamic_robots()
        
        # Manejo de energía
        self.energy -= 1
        if self.energy <= 0:
            done = True

        obs = np.array([*self.robot_position, int(self.item_picked), self.energy], dtype=np.int32)
        return obs, reward, done, {}

    def move_dynamic_robots(self):
        new_positions = []
        for robot in self.dynamic_robots:
            x, y = robot
            action = random.choice([0, 1, 2, 3])
            if action == 0 and y > 0:
                y -= 1
            elif action == 1 and y < self.grid_size[1] - 1:
                y += 1
            elif action == 2 and x > 0:
                x -= 1
            elif action == 3 and x < self.grid_size[0] - 1:
                x += 1
            new_positions.append((x, y))
        self.dynamic_robots = new_positions

# Configuración Q-Learning
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 1000

# Inicializar Q-Table (x, y, item, energy, actions)
q_table = np.zeros((10, 10, 2, 101, 4))

# Entrenamiento
env = WarehouseEnv()
rewards = []
deliveries = []
collisions = []
energy_usage = []

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        x, y, item, energy = state
        energy_idx = min(energy, 100)  # Limitar índice a 100
        
        # Selección de acción
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[x, y, item, energy_idx, :])
        
        # Ejecutar acción
        next_state, reward, done, _ = env.step(action)
        nx, ny, nitem, nenergy = next_state
        nenergy_idx = min(nenergy, 100)
        
        # Actualizar Q-Table
        current_q = q_table[x, y, item, energy_idx, action]
        max_future_q = np.max(q_table[nx, ny, nitem, nenergy_idx, :])
        new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
        q_table[x, y, item, energy_idx, action] = new_q
        
        state = next_state
        total_reward += reward
        
        if reward == 10:  # Entrega exitosa
            deliveries.append(1)
    
    # Decaimiento de epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)
    collisions.append(env.collisions)
    energy_usage.append(100 - env.energy)
    
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards[-100:])
        print(f"Episodio {episode+1}: Recompensa Promedio = {avg_reward:.2f}")

# Visualización del aprendizaje
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(rewards)
plt.title("Recompensas por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa")

plt.subplot(1, 3, 2)
plt.plot(np.cumsum(rewards))  # Recompensas acumuladas
plt.title("Recompensas Acumuladas")
plt.xlabel("Episodio")
plt.ylabel("Recompensa Acumulada")

plt.subplot(1, 3, 3)
plt.plot(np.cumsum(deliveries), color='orange')
plt.title("Entregas Acumuladas")
plt.xlabel("Episodio")
plt.ylabel("Total Entregas")
plt.tight_layout()
plt.show()

# Gráficos adicionales
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(collisions)
plt.title("Colisiones por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Colisiones")

plt.subplot(1, 2, 2)
plt.plot(energy_usage)
plt.title("Uso de Energía por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Energía Restante")
plt.tight_layout()
plt.show()

# Comparación con método heurístico
def heuristic_agent(env):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        x, y, item, energy = state
        
        # Heurística simple: ir al punto de recogida, luego al de entrega
        if not item:
            if x < env.pickup_point[0]:
                action = 3
            elif x > env.pickup_point[0]:
                action = 2
            elif y < env.pickup_point[1]:
                action = 1
            elif y > env.pickup_point[1]:
                action = 0
            else:
                item = True
        else:
            if x < env.delivery_point[0]:
                action = 3
            elif x > env.delivery_point[0]:
                action = 2
            elif y < env.delivery_point[1]:
                action = 1
            elif y > env.delivery_point[1]:
                action = 0
        
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    
    return total_reward

heuristic_rewards = []
for _ in range(100):
    env = WarehouseEnv()
    heuristic_rewards.append(heuristic_agent(env))

print(f"Recompensa promedio heurística: {np.mean(heuristic_rewards)}")

# Visualización final
def visualize_trained(env, q_table):
    pygame.init()
    WIDTH, HEIGHT = 600, 600
    cell_size = WIDTH // env.grid_size[0]
    colors = {
        'background': (255, 255, 255),
        'robot': (41, 128, 185),
        'obstacle': (142, 68, 173),
        'pickup': (39, 174, 96),
        'delivery': (231, 76, 60),
        'dynamic_robot': (241, 196, 15)
    }
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Agente Entrenado")
    font = pygame.font.Font(None, 30)
    
    state = env.reset()
    done = False
    clock = pygame.time.Clock()
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        screen.fill(colors['background'])
        
        # Dibujar grid
        for x in range(env.grid_size[0]):
            for y in range(env.grid_size[1]):
                rect = pygame.Rect(x*cell_size, y*cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, (200, 200, 200), rect, 1)
        
        # Dibujar elementos
        for obstacle in env.obstacles:
            pygame.draw.rect(screen, colors['obstacle'], 
                           (obstacle[0]*cell_size + 2, obstacle[1]*cell_size + 2, cell_size-4, cell_size-4))
        
        pygame.draw.rect(screen, colors['pickup'], 
                       (env.pickup_point[0]*cell_size + 2, env.pickup_point[1]*cell_size + 2, cell_size-4, cell_size-4))
        pygame.draw.rect(screen, colors['delivery'], 
                       (env.delivery_point[0]*cell_size + 2, env.delivery_point[1]*cell_size + 2, cell_size-4, cell_size-4))
        
        for robot in env.dynamic_robots:
            pygame.draw.circle(screen, colors['dynamic_robot'], 
                             (robot[0]*cell_size + cell_size//2, robot[1]*cell_size + cell_size//2), cell_size//3)
        
        # Dibujar robot principal
        robot_color = colors['robot'] if env.item_picked else (189, 195, 199)
        pygame.draw.circle(screen, robot_color, 
                         (env.robot_position[0]*cell_size + cell_size//2, 
                          env.robot_position[1]*cell_size + cell_size//2), cell_size//2 - 2)
        
        # Mostrar estadísticas
        stats = f"Entregas: {env.deliveries} | Energía: {env.energy}"
        text = font.render(stats, True, (0, 0, 0))
        screen.blit(text, (10, 10))
        
        pygame.display.flip()
        clock.tick(10)
        
        # Seleccionar acción usando Q-Table
        x, y = env.robot_position
        item = int(env.item_picked)
        energy_idx = min(env.energy, 100)
        action = np.argmax(q_table[x, y, item, energy_idx, :])
        
        _, _, done, _ = env.step(action)
        
    pygame.quit()

# Ejecutar visualización final
visualize_trained(env, q_table)
