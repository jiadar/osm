"""
from functools import wraps
import time
from typing import Any
import itertools
import json
import math
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
    weight: float = 0


@dataclasses.dataclass(order=True)
class Waypoint:
    distance: float
    vertex: Any = dataclasses.field(compare=False)


#    via: Any = dataclasses.field(compare=False)


@dataclasses.dataclass(order=True)
class Distance:
    distance: float
    vertex: Any = dataclasses.field(compare=False)


@dataclasses.dataclass
class Segment:
    p1: Any
    p2: Any
    lineq: Any = None
    m: float = 0
    c: float = 0
    eqn: str = ""
    dv: str = "x"


@dataclasses.dataclass
class Line:
    start: Any
    end: Any
    lineq: Any = None
    eqn: str = ""
    m: float = 0
    dv: str = "x"


@dataclasses.dataclass
class SimpleLine:
    start: Any
    end: Any
    eqn: Any = None
    m: Any = None


class Graph:
    data = []
    lines = []
    graph = {}
    mingraph = {}
    degrees = {}
    degree_map = {}
    segments = []
    segment_map = {}
    eqns_dict = {}

    # Now 0.7 ms
    #
    # This doesn't deal properly with non-connected graphs. Need to do: Group the graphs into groups of connected
    # components, then run this algorithm over each set of connected components.
    #
    @timeit
    def __init__(self, data):
        self.data = data
        self.lines = [[(item[0], item[1]) for item in line] for line in data]

        # 0.2 ms
        def add_lines_to_graph():
            # Add the lines to the graph
            for line in self.lines:
                for idx in range(len(line)):
                    # Add the vertex using the current index, the first element is not connected to anything yet
                    self.add_vertex(line[idx])
                    # Add the edge. If we have more than 1 element, there is an edge connecting to the former element
                    if idx > 0:
                        edge = (line[idx - 1], line[idx])
                        self.add_edge(edge)

                # Traverse the line in reverse and add the edges, since the graph is undirected we must have edges
                # pointing both ways to route
                line.reverse()
                for idx in range(len(line)):
                    # Add the vertex using the current index, the first element is not connected to anything yet
                    self.add_vertex(line[idx])
                    # Add the edge. If we have more than 1 element, there is an edge connecting to the former element
                    if idx > 0:
                        edge = (line[idx - 1], line[idx])
                        self.add_edge(edge)

        add_lines_to_graph()

        # 0.2 ms
        def add_edges_to_graph():
            for vertex, edges in self.graph.items():
                for edge in edges:
                    (x1, y1) = vertex
                    (x2, y2) = edge.vertex
                    segment = Segment(p1=vertex, p2=edge.vertex, m=edge.m)
                    if self.segment_map.get(edge.m):
                        self.segment_map[edge.m].append(segment)
                    else:
                        self.segment_map[edge.m] = [segment]
                    self.segments.append(segment)

        add_edges_to_graph()

        def make_segment_map():
            # Go through each slope and make linear equation to match all segments possible on that line
            for m in self.segment_map.keys():
                for segment in self.segment_map[m]:
                    (x1, y1) = segment.p1
                    if m == 0:
                        segment.eqn = f"y = {y1}"
                        segment.c = 0
                        segment.dv = "x"
                        segment.lineq = (0, 1, y1)
                    elif m < float("inf"):
                        c = y1 + m * x1 * -1
                        sign = "-" if c < 0 else "+"
                        segment.eqn = f"y = {m}x {sign} {abs(c)}"
                        segment.c = c
                        segment.dv = "x"
                        segment.lineq = (-1 * m, 1, c)
                    else:
                        segment.eqn = f"x = {x1}"
                        segment.c = 0
                        segment.dv = "y"
                        segment.lineq = (1, 0, x1)

        make_segment_map()

        def eqns_from_segments():
            self.eqns = [
                segment
                for segment_lst in self.segment_map.values()
                for segment in segment_lst
            ]

        self.eqns = eqns_from_segments()

        # Finally, make the equations and ranges

        # 0.06 ms
        def make_ranges():
            for segment in self.segments:
                if self.eqns_dict.get(segment.eqn):
                    self.eqns_dict[segment.eqn].append(
                        Line(
                            start=segment.p1,
                            end=segment.p2,
                            eqn=segment.eqn,
                            m=segment.m,
                            dv=segment.dv,
                            lineq=segment.lineq,
                        )
                    )
                else:
                    self.eqns_dict[segment.eqn] = [
                        Line(
                            start=segment.p1,
                            end=segment.p2,
                            eqn=segment.eqn,
                            m=segment.m,
                            dv=segment.dv,
                            lineq=segment.lineq,
                        )
                    ]

        make_ranges()

        def get_intersection(l1, l2):
            a1, b1, c1 = l1
            a2, b2, c2 = l2
            det = a1 * b2 - a2 * b1
            if det == 0:
                return None
            x = (c1 * b2 - c2 * b1) / det
            y = (a1 * c2 - a2 * c1) / det
            return (x, y)

        # 0.2 ms
        self.final = []

        def make_final():
            for eq in self.eqns_dict.keys():
                start_range = None
                end_range = None
                for zl in self.eqns_dict[eq]:
                    if start_range is None:
                        if zl.dv == "y":
                            start_range = min(zl.start[1], zl.end[1])
                            end_range = max(zl.start[1], zl.end[1])
                        else:
                            start_range = min(zl.start[0], zl.end[0])
                            end_range = max(zl.start[0], zl.end[0])
                    else:
                        if zl.dv == "y":
                            start_range = min([zl.start[1], zl.end[1], start_range])
                            end_range = max([zl.start[1], zl.end[1], end_range])
                        else:
                            start_range = min([zl.start[0], zl.end[0], start_range])
                            end_range = max([zl.start[0], zl.end[0], end_range])
                soln = None
                if zl.dv == "x":
                    x = start_range
                    a, b, c = zl.lineq
                    soln = (c - x * a) / b
                    start_pt = (x, soln)
                    x = end_range
                    soln = (c - x * a) / b
                    end_pt = (x, soln)
                else:
                    y = start_range
                    a, b, c = zl.lineq
                    soln = (c - y * b) / a
                    start_pt = (soln, y)
                    y = end_range
                    soln = (c - y * b) / a
                    end_pt = (soln, y)
                self.final.append(
                    SimpleLine(start=start_pt, end=end_pt, eqn=zl.lineq, m=zl.m)
                )
                start_range = None
                end_range = None

        make_final()
        # 0.02 ms

        # 0.12 ms
        def get_intersection_set():
            intersection_set = set()
            for pql in itertools.combinations(self.final, 2):
                c1, c2 = pql
                isc = get_intersection(c1.eqn, c2.eqn)
                if isc:
                    x, y = isc
                    x1, y1 = pql[0].start
                    x2, y2 = pql[0].end
                    x3, y3 = pql[1].start
                    x4, y4 = pql[1].end
                    within_bounds = (
                        x >= min(x1, x2)
                        and x <= max(x1, x2)
                        and y >= min(y1, y2)
                        and y <= max(y1, y2)
                        and x >= min(x3, x4)
                        and x <= max(x3, x4)
                        and y >= min(y3, y4)
                        and y <= max(y3, y4)
                    )
                    if within_bounds:
                        intersection_set.add(isc)
            return list(intersection_set)

        self.intersections = get_intersection_set()

    def vertices(self):
        return list(self.graph.keys())

    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []
            self.degrees[vertex] = 0

    def distance_between(self, vertex1, vertex2):
        return round(math.dist([vertex1[0], vertex1[1]], [vertex2[0], vertex2[1]]), 2)

    def slope(self, vertex1, vertex2):
        (x1, y1) = vertex1
        (x2, y2) = vertex2
        if x1 == x2:
            return float("inf")
        if y1 == y2:
            return 0
        return round((y2 - y1) / (x2 - x1), 3)

    def add_edge(self, edge):
        if len(edge) < 2:
            return
        (vertex1, vertex2) = edge
        weight = self.distance_between(vertex1, vertex2)
        m = self.slope(vertex1, vertex2)
        edge_to_insert = Edge(vertex=vertex2, weight=weight, m=m)
        if vertex1 in self.graph:
            if edge_to_insert not in self.graph[vertex1] and vertex1 != vertex2:
                self.graph[vertex1].append(edge_to_insert)
                self.degrees[vertex1] += 1
        else:
            self.graph[vertex1] = [edge_to_insert]
            self.degrees[vertex1] = 1

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
        for sl in self.final:
            xs = [sl.start[0], sl.end[0]]
            ys = [sl.start[1], sl.end[1]]
            plt.plot(xs, ys, c="gray")
            plt.plot(sl.start[0], sl.start[1], "o", c="blue")
            plt.plot(sl.end[0], sl.end[1], "o", c="blue")
        for i in self.intersections:
            plt.plot(i[0], i[1], "o", c="blue")

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
    pprint(topo.final)
    # pprint(topo.segment_map)
    # topo.clean_data2(0.5)
    # print("\nGraph:\n")
    # print(topo.to_str())

    # path = topo.compute_weighted_path((4, 1), (1, 4))
    # print("\nCalculated path from (4,1) to (1,4): \n")
    # pprint.pprint(path)

    # plot(data, path)
    topo.plot().show()
"""
