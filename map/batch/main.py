from functools import wraps
import time
from typing import Any
import json
import matplotlib.pyplot as plt
from pprint import pprint
from queue import PriorityQueue, Queue
import dataclasses

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
    with open("/Users/javin/work/osm/data/data2.json") as file:
        data = json.load(file)
    return data


@dataclasses.dataclass
class Edge:
    vertex: Any
    m: float = 0
    weight: float = 1


@dataclasses.dataclass(order=True)
class Waypoint:
    distance: float
    vertex: Any = dataclasses.field(compare=False)


#    via: Any = dataclasses.field(compare=False)


@dataclasses.dataclass(order=True)
class Distance:
    distance: float
    vertex: Any = dataclasses.field(compare=False)


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
                    edge = (line[idx - 1], line[idx])
                    self.add_edge(edge)

            line.reverse()

            for idx in range(len(line)):
                self.add_vertex(line[idx])
                if idx > 0:
                    edge = (line[idx - 1], line[idx])
                    self.add_edge(edge)

    @timeit
    def __init__(self, data):
        lines = [[(item[0], item[1]) for item in line] for line in data]
        self.__add_lines_to_graph(lines)

    def vertices(self):
        return list(self.graph.keys())

    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []

    def add_edge(self, edge):
        if len(edge) < 2:
            return
        (vertex1, vertex2) = edge
        edge_to_insert = Edge(vertex=vertex2, weight=1)
        if vertex1 in self.graph:
            if edge_to_insert not in self.graph[vertex1] and vertex1 != vertex2:
                self.graph[vertex1].append(edge_to_insert)
        else:
            self.graph[vertex1] = [edge_to_insert]

    def to_str(self):
        ordered_graph = sorted(self.graph.items(), key=lambda e: (e[0][0], e[0][1]))
        p = ""
        for adj_list in ordered_graph:
            p = p + f"{adj_list[0]} -> "
            for neighbor in adj_list[1]:
                p = p + f"{neighbor}, "
            p = p[:-2] + "\n"
        return p

    def compute_weighted_path(self, start, goal):
        q = []
        for v in self.graph.keys():
            if v == start:
                q.append(Waypoint(distance=0.0, vertex=start))
            else:
                q.append(Waypoint(distance=99.0, vertex=v))
        finished = Queue()
        i = 0
        DEBUG = False
        print()
        while len(q) > 0 and i < 99 and q[0].vertex != goal:
            if i == 4:
                DEBUG = False
            q.sort(key=lambda e: e.distance)
            u = q.pop(0)
            print(f"Processing: {u}")
            DEBUG and print(f"Adj: {self.graph[u.vertex]}")
            for v in self.graph[u.vertex]:
                # Check if v is in min heap
                for idx, wp in enumerate(q):
                    if wp.vertex == v.vertex:
                        # if v is in min heap, and distance value is more than weight of u-v plus distance value of u
                        DEBUG and print(
                            f"Found: {wp.vertex} in heap with distance {wp.distance}"
                        )
                        this_distance = u.distance + v.weight
                        DEBUG and print(
                            f"Distance: {u.distance} + {v.weight} = {this_distance}"
                        )
                        if wp.distance > this_distance:
                            DEBUG and print(f"Update: {wp.distance} > {this_distance}")
                            # update the distance value of v
                            new_wp = Waypoint(distance=this_distance, vertex=v.vertex)
                            DEBUG and print(f"NewWp: {new_wp}")
                            q[idx] = new_wp
                        else:
                            DEBUG and print(
                                f"NoUpdate: {wp.distance} <= {this_distance}"
                            )
            i = i + 1

        return q[0].distance

    def compute_weighted_path2(self, start, goal):
        q = PriorityQueue()
        for v in self.graph.keys():
            if v == start:
                q.put(Waypoint(priority=0.0, vertex=start, via=None))
            else:
                q.put(Waypoint(priority=99.0, vertex=v, via=None))
        finished = Queue()
        i = 0
        cur_wp = q.get()
        while cur_wp.vertex != goal and not q.empty():
            for neighbor in self.graph[cur_wp.vertex]:
                next_priority = neighbor.priority + cur_wp.priority
                next_wp = Waypoint(
                    priority=next_priority, vertex=neighbor.vertex, via=cur_wp.vertex
                )
                q.put(next_wp)
            finished.put(cur_wp)
            cur_wp = q.get()
            i = i + 1

        print("PriorityQueue: ")
        while not q.empty():
            print(q.get())
        print("Finished Pile: ")
        pprint.pprint(finished.queue)
        print()

        return finished

    def plot(self, path=None):
        plt.figure(figsize=(16, 10))
        plt.subplot(2, 5, 1)
        plt.axis(axis_coords)
        colors = ["b", "g", "r", "k", "m", "y", "olive", "c"]
        for vertex, edges in self.graph.items():
            for edge in edges:
                x_coords = [vertex[0], edge.vertex[0]]
                y_coords = [vertex[1], edge.vertex[1]]
                plt.plot(x_coords, y_coords, c="gray")
            plt.plot(vertex[0], vertex[1], "o", c="blue")

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
    #pprint(topo.final)
    # pprint(topo.segment_map)
    # topo.clean_data2(0.5)
    # print("\nGraph:\n")
    # print(topo.to_str())

    # path = topo.compute_weighted_path((4, 1), (1, 4))
    # print("\nCalculated path from (4,1) to (1,4): \n")
    # pprint.pprint(path)

    # plot(data, path)
    topo.plot().show()
