from random import randint
class snake:
    def __init__(self,x,y,X,Y):
        self.X = X
        self.Y = Y
        self.dx = 0
        self.dy = 0
        self.direction = -1
        self.body = [(x,y)]
    def move(self, direction, amount):
        if direction == self.direction or (direction+self.direction)%2 == 0:
            return
        if direction == 0:
            self.dy = -amount
            self.dx = 0
        elif direction == 1:
            self.dx = amount
            self.dy = 0
        elif direction == 2:
            self.dy = amount
            self.dx = 0
        elif direction == 3:
            self.dx = -amount
            self.dy = 0
        else:
            return
        self.direction = direction
    def getPos(self):
        return self.body[-1][0], self.body[0][1]
    def getMovement(self):
        return self.dx, self.dy
    def die(self):
        return len(self.body) != len(set(self.body))
    def eats(self, apple):
        if (self.body[-1][0] + self.dx)%self.X == apple[0] and (self.body[-1][1]+self.dy)%self.Y == apple[1]:
            self.body += [apple]
#             print(self.body)
            return True
        return False
    
import pygame
pygame.init()
X = 400
Y = 300
gameDisplay = pygame.display.set_mode((X,Y))
pygame.display.set_caption(u"貪食蛇")

red = (255,0,0)
green = (0,255,0)
black = (0,0,0)

x = X//2
y = Y//2
squareSide = moveAmount = 10
score = 0

myfont = pygame.font.Font('C:\Windows\Fonts\msjh.ttc', 20)

clock = pygame.time.Clock()

snake = snake(x,y,X,Y)
apples = [(randint(0, X//moveAmount-1)*10, randint(0, Y//moveAmount-1)*10) for i in range(10)]
# print(apples)

gameExit = False
while not gameExit:
    gameDisplay.fill((255,255,255))
    gameDisplay.lock()
    for apple in apples:
        if snake.eats(apple):
            apples.remove(apple)
            score += 1
            if len(apples) == 0:
                apples = [(randint(0, X//moveAmount-1)*10, randint(0, Y//moveAmount-1)*10) for i in range(10)]
        gameDisplay.fill(green, [apple[0],apple[1],squareSide,squareSide])
    gameDisplay.unlock()
    gameDisplay.blit(myfont.render(u"蛇", True, red), [x,y])
    gameDisplay.blit(myfont.render(str(score), True, red), [X//2,Y//2])
    for i in range(0,len(snake.body)-1):
        body = snake.body[i] = snake.body[i+1]
        gameDisplay.fill(black, [body[0],body[1],squareSide,squareSide])
    x = (snake.body[-1][0]+snake.dx)%X
    y = (snake.body[-1][1]+snake.dy)%Y
    snake.body[-1] = (x,y)
    gameDisplay.fill(red, [snake.body[-1][0],snake.body[-1][1],squareSide,squareSide])
    
    if snake.die():
        gameDisplay.blit(myfont.render(u"你死了", True, red), [0,0])
        pygame.display.update()
        pygame.time.wait(2000)
        gameExit = True
    
    pygame.display.update()
    clock.tick(20)
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            gameExit = True
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_LEFT:
                snake.move(3, moveAmount)
            elif e.key == pygame.K_RIGHT:
                snake.move(1, moveAmount)
            elif e.key == pygame.K_UP:
                snake.move(0, moveAmount)
            elif e.key == pygame.K_DOWN:
                snake.move(2, moveAmount)
pygame.quit()