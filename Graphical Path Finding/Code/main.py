import pygame
from queue import PriorityQueue
from lxml import etree
import time
import math

WIDTH = 720
DELAY_S = 0.1

WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("In search of battery")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)


class Cell:
    rows = 0
    cols = 0

    def __init__(self, row, col, width, height):
        self.row = row
        self.col = col
        self.x = col * width
        self.y = row * height
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.height = height

    def get_pos(self):
        return self.row, self.col

    def is_obstacle(self):
        return self.color == BLACK

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_obstacle(self):
        self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.height))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < Cell.rows - 1 and not grid[self.row + 1][self.col].is_obstacle():
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.col < Cell.cols - 1 and not grid[self.row][self.col + 1].is_obstacle():
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.row > 0 and not grid[self.row - 1][self.col].is_obstacle():
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col > 0 and not grid[self.row][self.col - 1].is_obstacle():
            self.neighbors.append(grid[self.row][self.col - 1])


def print_path(node_to_parent, current, updater):
    counter = 0
    while current in node_to_parent:
        current = node_to_parent[current]
        current.make_path()
        counter += 1
        updater()
    print(f'Path length: {counter}')


def find_path_breadth_first_search(updater, grid, start, end):
    counter = 0
    frontier = []
    node_to_parent = {}
    explored = set()

    frontier.append(start)
    if start == end:
        end.make_end()
        return True

    while len(frontier) > 0:
        time.sleep(DELAY_S)

        current = frontier[0]
        frontier = frontier[1:]
        explored.add(current)
        counter += 1

        for neighbor in current.neighbors:
            if neighbor not in explored and neighbor not in frontier:
                if neighbor == end:
                    node_to_parent[neighbor] = current
                    print_path(node_to_parent, neighbor, updater)
                    neighbor.make_end()
                    return True
                frontier.append(neighbor)
                node_to_parent[neighbor] = current
                neighbor.make_open()

        updater()

        if current != start:
            current.make_closed()

    return False


def find_path_UCS(updater, grid, start, end):
    counter = 0
    frontier = PriorityQueue()
    node_to_parent = {}

    f = {cell: math.inf for row in grid for cell in row}
    f[start] = 0

    frontier.put((0, counter, start))
    frontier_temp_set = {start}

    while not frontier.empty():
        time.sleep(DELAY_S)

        current = frontier.get()[2]
        frontier_temp_set.remove(current)

        if current == end:
            print_path(node_to_parent, current, updater)
            current.make_end()
            return True

        for neighbor in current.neighbors:
            if f[current] + 1 < f[neighbor]:
                node_to_parent[neighbor] = current
                f[neighbor] = f[current] + 1
                if neighbor not in frontier_temp_set:
                    counter += 1
                    frontier.put((f[neighbor], counter, neighbor))
                    frontier_temp_set.add(neighbor)
                    neighbor.make_open()

        updater()

        if current != start:
            current.make_closed()

    return False


def manhattan_heuristic(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def find_path_a_star(updater, grid, start, end):
    front_counter = 0
    exp_counter = 0
    frontier = PriorityQueue()
    node_to_parent = {}

    g = {cell: math.inf for row in grid for cell in row}
    g[start] = 0
    f = {cell: math.inf for row in grid for cell in row}
    f[start] = manhattan_heuristic(start.get_pos(), end.get_pos())

    frontier.put((0, front_counter, start))
    frontier_temp_set = {start}

    while not frontier.empty():
        time.sleep(DELAY_S)

        current = frontier.get()[2]
        frontier_temp_set.remove(current)

        if current == end:
            print_path(node_to_parent, current, updater)
            current.make_end()
            print(f'num of added to frontier: {front_counter}')
            print(f'num of explored: {exp_counter+1}')
            return True

        for neighbor in current.neighbors:
            if g[current] + 1 < g[neighbor]:
                node_to_parent[neighbor] = current
                g[neighbor] = g[current] + 1
                f[neighbor] = g[neighbor] + manhattan_heuristic(neighbor.get_pos(), end.get_pos())
                if neighbor not in frontier_temp_set:
                    front_counter += 1
                    frontier.put((f[neighbor], front_counter, neighbor))
                    frontier_temp_set.add(neighbor)
                    neighbor.make_open()

        updater()

        if current != start:
            exp_counter += 1
            current.make_closed()

    return False


def make_grid(room, rows, cols):
    grid = []
    x_gap = WIDTH // cols
    y_gap = WIDTH // rows

    robot = None
    battery = None

    for i in range(rows):
        grid.append([])
        for j in range(cols):
            cell = Cell(i, j, x_gap, y_gap)
            if room[i][j] == 'robot':
                robot = cell
                robot.make_start()

            elif room[i][j] == 'Battery':
                battery = cell
                battery.make_end()

            elif room[i][j] == 'obstacle':
                cell.make_obstacle()

            grid[i].append(cell)

    Cell.rows = rows
    Cell.cols = cols
    assert (robot and battery), 'Robot and battery must be given in sample input.'

    return grid, robot, battery


def update_win(win, grid, rows, cols):
    win.fill(WHITE)

    for row in grid:
        for cell in row:
            cell.draw(win)

    x_gap = WIDTH // cols
    y_gap = WIDTH // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * y_gap), (WIDTH, i * y_gap))
        for j in range(cols):
            pygame.draw.line(win, GREY, (j * x_gap, 0), (j * x_gap, WIDTH))
    pygame.display.update()


def parse_input_xml(input_path: str = 'SampleRoom.xml'):
    with open(input_path, 'rb') as f:
        xml = f.read()

    root = etree.XML(xml)
    room = []
    for row in root.getchildren():
        x = []

        for cell in row.getchildren():
            if not cell.text:
                text = 'empty'
            else:
                text = cell.text
            x.append(text)

        room.append(x)

    return room


if __name__ == '__main__':
    room = parse_input_xml()
    height, width = len(room), len(room[0])
    assert not any([abs(len(x) - width) for x in room]), 'XML rows must have same size.'

    grid, robot, battery = make_grid(room, height, width)

    run = True
    while run:
        update_win(WIN, grid, height, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    for row in grid:
                        for cell in row:
                            cell.update_neighbors(grid)

                    find_path_a_star(lambda: update_win(WIN, grid, height, width), grid, robot, battery)

                elif event.key == pygame.K_b:
                    for row in grid:
                        for cell in row:
                            cell.update_neighbors(grid)

                    find_path_UCS(lambda: update_win(WIN, grid, height, width), grid, robot, battery)

                elif event.key == pygame.K_f:
                    for row in grid:
                        for cell in row:
                            cell.update_neighbors(grid)

                    find_path_breadth_first_search(lambda: update_win(WIN, grid, height, width), grid, robot, battery)

                elif event.key == pygame.K_r:
                    grid, robot, battery = make_grid(room, height, width)

    pygame.quit()
