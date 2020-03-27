# %%

from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication, QMainWindow

# %%
from itertools import product
from functools import reduce
from plotly.graph_objects import Figure, Scatter
import plotly
import numpy as np
import string
from networkplotter import NetworkXPlotter

# %%

import networkx as nx
import plotly.graph_objects as go


# %%

class Node(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.connections = dict()

    def set_connections(self, connections):
        for connection in connections:
            self.connections[connection[1]] = connection[0]

    def set_weight(self, connection):
        self.connections[connection[1]] = connection[0]

    def set_value(self, value):
        self.value = value

    def remove_connection(self, node):
        try:
            self.connections.pop(node)
        except KeyError:
            print("No connection detected")

    def __str__(self):
        return self.name


class Graph(object):
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.A = []

    def form_subgraphs(self, clusters):
        clusters_keys = list(clusters.keys())
        clusters_values = list(clusters.values())
        subgraphs = []
        connections = []

        for cluster in clusters_keys:
            value = reduce(lambda x, y: x.value + y.value, clusters[cluster]) \
                if len(clusters[cluster]) > 1 else clusters[cluster][0].value

            subgraphs.append(Node(cluster, value))

        for i in range(len(clusters)):
            connections = []
            for j in range(len(clusters)):
                weight = 0
                for node_left in clusters_values[i]:
                    for node_right in node_left.connections.keys():
                        additive = node_left.connections[node_right] if node_right in clusters_values[j] \
                                                                        and not node_right in clusters_values[i] else 0
                        weight += additive
                connections.append([weight, subgraphs[j]])

            subgraphs[i].set_connections(connections)

        aggregated = Graph('aggregated')
        aggregated.set_nodes(subgraphs)
        return aggregated

    def set_nodes(self, nodes):
        self.nodes = nodes

    def add_node(self, node):
        self.nodes.append(node)

    def remove_node(self, node):
        try:
            self.nodes.remove(node)
        except ValueError:
            print("no node " + str(node) + " in the graph")

    def __str__(self):
        return self.name

    def form_connection_matrix(self):
        self.A = np.zeros((len(self.nodes), len(self.nodes)))
        for i in range(len(self.nodes)):

            connection_nodes = list(self.nodes[i].connections.keys())
            for j in range(len(self.nodes)):
                self.A[i, j] = self.nodes[i].connections[self.nodes[j]] \
                    if self.nodes[j] in connection_nodes else 0

        return self.A

    def update_values(self, nodes_values):
        for node in self.nodes:
            node.set_value(nodes_values[node])

    def send_impulses(self, impulses):
        previous = np.zeros((len(self.nodes), 1))
        current = np.array([[node.value] for node in self.nodes])
        for impulse in impulses:
            new = current + np.array(self.A).T @ (current - previous) + impulse
            nodes_values = {self.nodes[i]: current[i, 0] for i in range(len(self.nodes))}
            previous = current
            current = new
            self.update_values(nodes_values)
        print(current)

    def search_cycles(self):
        cycles = set()
        for node in self.nodes:
            print(self.nodes.index(node))
            layer = 0
            vertexes = dict()
            vertexes[layer] = {node}
            while layer < len(self.nodes):
                nexts = set()
                previous_layers = reduce(lambda x, y: x.union(y), list(vertexes.values())[: layer]) if layer > 0 else []
                for vertex in vertexes[layer]:
                    if vertex not in previous_layers:
                        for next_vertex in vertex.connections.keys():
                            nexts.add(next_vertex)
                    else:
                        nexts.add(vertex)
                layer += 1
                if nexts != set():
                    vertexes[layer] = nexts
                else:
                    break

                    # for i in range(layer):
            #    print(list(map(lambda val: str(val), vertexes[i])))
            # print("----------")
            for length in range(1, len(vertexes)):
                if node in vertexes[length]:
                    additive = list(product(*[vertexes[i] for i in range(0, length)]))
                    for cycle in additive:
                        cycles.add(tuple(set(cycle)))

        for cycle in cycles:
            print(list(map(lambda val: str(val), cycle)))
            print("----------")
            # cycle = set()
            # for layer in range(1, length):
            #    cycle.add()
            # for node in nod


# %%

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        likelyhood = 0.06
        nods = [
            Node(letter, number)
            for letter, number in zip(
                string.ascii_lowercase, range(len(string.ascii_lowercase) - 10)
            )
        ]
        for nod in nods:
            nod.set_connections(
                [
                    [rand, x]
                    for rand, x in zip(np.random.uniform(0, 1, len(nods)), nods)
                    if np.random.binomial(1, likelyhood)
                ]
            )

        example_network = Graph("Example")
        example_network.set_nodes(nods)
        example_network.form_connection_matrix()

        netplot = NetworkXPlotter(example_network, layout="spectral")
        netplot.plot(
            colorscale="sunset",
            edge_opacity=0.6,
            edge_color="SlateGrey",
            node_opacity=1,
            node_size=12,
            edge_scale=3,
            title="Cognitive network visualization<br>(wider edges have higher weights, bigger nodes have self edge)",
            fontsize=12,
            plot_text=True,
        )
        # we create an instance of QWebEngineView and set the html code
        plot_widget = QWebEngineView()
        plot_widget.setHtml(netplot.html)
        # set the QWebEngineView instance as main widget
        self.setCentralWidget(plot_widget)


# %%

app = QApplication([])
window = MainWindow()
window.show()
app.exec_()
