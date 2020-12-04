from functools import wraps
import time
from typing import Any, Tuple
import json
import matplotlib.pyplot as plt
from pprint import pprint
import queue
from dataclasses import dataclass

plt.style.use("seaborn-whitegrid")
axis_coords = [0, 6, 0, 6]

# Questions:
# I'd like to clarify the following:
# Memory available, Allowed wait time, size of input data, inconsistencies in data, other attributes on data we would consider
# in routing (cost, etc), delta to use in identifying nearby points, number of concurrent requests per second, expected response time
# to user.
#
# If I had more time:
# I'd finish the fuzzy path search, write tests with varying data, write a mini visualization/chart, organize the code better into modules,
# generate performance testing data, set up a message queue that would dispatch expensive computations so the UI won't hang, add error messages,
# validate data before processing, fix up variable names so the code is more readable, add types and explinations of how to use the data
# structures and graph, etc.
#
# Q3:
# See the compute path method, the answer is in line.


def timeit(my_func):
    @wraps(my_func)
    def timed(*args, **kw):

        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()

        print(
            '"{}" took {:.3f} ms to execute\n'.format(
                my_func.__name__, (tend - tstart) * 1000
            )
        )
        return output

    return timed


def get_data():
    with open("/Users/javin/work/osm/data/simpledata.json") as file:
        data = json.load(file)
    return data


Vertex = Tuple[float, float]


@dataclass
class Edge:
    vertex: Vertex
    weight: float = 0


class Graph:
    data = []
    graph = {}
    lines = []

    def __add_lines_to_graph(self, lines):
        # Add the lines to the graph
        self.lines = lines
        for line in lines:
            for idx in range(len(line)):
                self.add_vertex(line[idx])
                if idx > 0:
                    vertex1, vertex2 = (line[idx - 1], line[idx])
                    self.add_edge(vertex1, vertex2)

            line.reverse()

            for idx in range(len(line)):
                self.add_vertex(line[idx])
                if idx > 0:
                    vertex1, vertex2 = (line[idx - 1], line[idx])
                    self.add_edge(vertex1, vertex2)

    @timeit
    def __init__(self, data):
        lines = [[(item[0], item[1]) for item in line] for line in data]
        self.__add_lines_to_graph(lines)

    def vertices(self):
        return list(self.graph.keys())

    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []

    def add_edge(self, vertex1, vertex2, weight=1):
        if vertex1 in self.graph:
            edge = Edge(vertex=vertex2, weight=weight)
            if vertex2 not in self.graph[vertex1] and vertex1 != vertex2:
                self.graph[vertex1].append(edge)
        else:
            self.graph[vertex1] = [edge]

    def to_str(self):
        ordered_graph = sorted(self.graph.items(), key=lambda e: (e[0][0], e[0][1]))
        p = ""
        for adj_list in ordered_graph:
            p = p + f"{adj_list[0]} -> "
            for neighbor in adj_list[1]:
                p = p + f"{neighbor}, "
            p = p[:-2] + "\n"
        return p

    def bfs(self, start, goal):
        q = queue.Queue()
        discovered = [start]
        q.put([start])
        while not q.empty():
            path = q.get()
            v = path[-1]
            if v == goal:
                return path
            for edge in self.graph[v]:
                if edge.vertex not in discovered:
                    discovered.append(edge.vertex)
                    next_path = list(path)
                    next_path.append(edge.vertex)
                    q.put(next_path)
        return []

    def plot(self, path=None):
        plt.figure(figsize=(16, 10))
        plt.subplot(2, 5, 1)
        plt.axis(axis_coords)
        colors = ["b", "g", "r", "k", "m", "y", "olive", "c"]
        for vertex1, edges in self.graph.items():
            for vertex2 in edges:
                x_coords = [vertex1[0], vertex2[0]]
                y_coords = [vertex1[1], vertex2[1]]
                plt.plot(x_coords, y_coords, c="gray")
            plt.plot(vertex1[0], vertex1[1], "o", c="blue")

        plt.subplot(2, 5, 2)
        plt.axis(axis_coords)

        for i, line in enumerate(self.lines):
            x_coords = [point[0] for point in line]
            y_coords = [point[1] for point in line]
            rgb = colors[i]
            plt.subplot(2, 5, i + 3)
            plt.axis(axis_coords)
            plt.plot(x_coords, y_coords, "o", c=rgb)
            plt.plot(x_coords, y_coords, c=rgb)

        return plt


if __name__ == "__main__":
    data = get_data()
    topo = Graph(data)
    # pprint(topo.graph)
    #pprint(topo.final)
    # pprint(topo.segment_map)
    # topo.clean_data2(0.5)
    # print("\nGraph:\n")
    # print(topo.to_str())

    # path = topo.compute_weighted_path((4, 1), (1, 4))
    # print("\nCalculated path from (4,1) to (1,4): \n")
    # pprint.pprint(path)

    # plot(data, path)
    # topo.plot().show()
    routes = {
        (1, 3): [(5, 5), (2, 4), (2, 2)],
        (2, 2): [(2, 4), (5, 5)],
        (2, 4): [(2, 2), (5, 5)]
    }
    paths = []
    for start, lst in routes.items():
        for end in lst:
            path = topo.bfs(start, end)
            paths.append(path)
    pprint(paths)
