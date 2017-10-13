import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import random
vertices= [(1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
           (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)]
vertices2= [tuple(map(sum, zip((2,2,2), v))) for v in vertices]
edges = [(0,1), (0,3), (0,4), (2,1), (2,3), (2,7),
    (6,3), (6,4), (6,7), (5,1), (5,4), (5,7)]
surfaces = [(0,1,2,3), (3,2,7,6), (6,7,5,4),
    (4,5,1,0), (1,5,7,2), (4,0,3,6)]
class Cube:
    def __init__(self, vertices, position, speed):
        self.vertices = vertices
        self.position = position
        self.speed = speed
    def draw(self):
        glPushMatrix()
        glTranslatef(*self.position)
        self.position = tuple(map(sum, zip(self.position, self.speed)))
        glBegin(GL_QUADS)
        for surface in surfaces:
            if surface == (1,5,7,2):
                glColor3fv((0,0,1))
            else:
                glColor3fv((0,1,0))
            for vertex in surface:
                glVertex3fv(self.vertices[vertex])
        glEnd()
        glBegin(GL_LINES)
        glColor3fv((1,0,0))
        for edge in edges:
            for vertex in edge:
                glVertex3fv(self.vertices[vertex])
        glEnd()
        glPopMatrix()
        
class Star:
    def __init__(self, radius, position, speed):
        self.radius = radius
        self.position = position
        self.speed = speed
    def draw(self):
        glPushMatrix()
        glTranslatef(*self.position)
        self.position = tuple(map(sum, zip(self.position, self.speed)))
        gluSphere(gluNewQuadric(), self.radius, 50, 50)
        glPopMatrix()
    def __str__(self):
        return str(self.position)+" "+str(self.speed)
        
pygame.init()
display = (640, 480)
screen = pygame.display.set_mode(display ,DOUBLEBUF|OPENGL)
gluPerspective(50, (display[0]/display[1]), 0.1, 50.0)
glTranslatef(0,0,-20)
# glRotatef(50, 1, 1, 0)
# c = Cube(vertices,(0,0,0),(0,0,0.2))
# c2 = Cube(vertices2,(0,0,0),(0,0,0.5))
stars = [Star(random.uniform(0,0.1),(random.uniform(-5,5),random.uniform(-5,5),random.uniform(-2,2)),
              (0,0,random.uniform(0.1,1))) for i in range(500)]

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                glTranslatef(0.5,0,0)
            if event.key == pygame.K_RIGHT:
                glTranslatef(-0.5,0,0)
            if event.key == pygame.K_UP:
                glTranslatef(0,-1,0)
            if event.key == pygame.K_DOWN:
                glTranslatef(0,1,0)
    mouseState = pygame.mouse.get_pressed()
    if mouseState[0]:
        glRotatef(1, 0, 0, 1)
    elif mouseState[2]:
        glRotatef(-1, 0, 0, 1)
    
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    for star in stars:
        if star.position[2] > 20:
            star.position = (random.uniform(-5,5),random.uniform(-5,5),random.uniform(-2,2))
        star.draw()
#     c.draw()
#     c2.draw()
#     if c.position[2] > 20:
#         c.position = (0,0,0)
#     if c2.position[2] > 20:
#         c2.position = (0,0,0)
    
    pygame.display.flip()
    pygame.time.wait(10)