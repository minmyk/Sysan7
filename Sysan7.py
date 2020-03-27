import numpy as np
import pandas as pd
from functools import reduce
from copy import copy
from PyQt5.QtWidgets import QApplication, QLineEdit, QDialog, \
    QHBoxLayout, QGridLayout, QStyleFactory, QCheckBox, QPushButton, QTabWidget, QTextEdit
from PyQt5.QtGui import QPalette, QColor, QIcon
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt
import sys


def show_qt(raw_html):
    fig_view = QWebEngineView()
    fig_view.setHtml(raw_html)
    # fig_view.show()
    fig_view.raise_()
    return fig_view


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


def form_sw_table(swot, t_count):
    sw = swot.index
    sw_table = np.zeros((len(sw), len(sw)))

    for i in range(len(sw) - 1):
        for j in range(i + 1, len(sw)):
            weight = 0
            for k in range(len(swot.iloc[i])):
                weight = weight - min(swot.iloc[i][k], swot.iloc[j][k]) if k < t_count \
                    else weight + min(swot.iloc[i][k], swot.iloc[j][k])

            sw_table[i, j] = round(weight, 5)

    sw_table = sw_table + sw_table.T
    print(sw_table[0, :])
    sw = pd.DataFrame({swot.index[i]: sw_table[:, i] for i in range(sw_table.shape[1])}, index=swot.index)
    return sw


def form_subgraphs(clusters):
    clusters_keys = list(clusters.keys())
    clusters_values = list(clusters.values())
    subgraphs = []

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
                    additive = node_left.connections[node_right] \
                        if node_right in clusters_values[j] \
                        and node_right not in clusters_values[i] else 0
                    weight += additive
            connections.append([weight, subgraphs[j]])

        subgraphs[i].set_connections(connections)

    aggregated = Graph('aggregated')
    aggregated.set_nodes(subgraphs)
    return aggregated


class Graph(object):
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.A = []

    def from_pandas(self, swot, sw, s_count):
        nodes = dict()

        for index in swot.index:
            value = 0
            row = swot.loc[index]
            for i in range(len(row)):
                value = value + row[i]
            nodes[index] = Node(index, value)

        for column in swot.columns:
            value = 0
            col = swot[column]
            for i in range(len(col)):
                value = value + col[i] if i < s_count else value - col[i]
            nodes[column] = Node(column, value)

        for index in swot.index:
            for column in swot.columns:
                if swot.loc[index][column] != 0:
                    nodes[index].set_weight([swot.loc[index][column], nodes[column]])
                    nodes[column].set_weight([swot.loc[index][column], nodes[index]])

        for index in sw.index:
            for column in sw.columns:
                nodes[index].set_weight([sw.loc[index][column], nodes[column]]) \
                    if sw.loc[index][column] != 0 else 0

        self.set_nodes(list(nodes.values()))

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

    def search_cycles(self, verbose=False):
        cycles = set()
        for node in self.nodes:
            layer = 0
            vertexes = dict()
            vertexes[layer] = {node}
            while True:
                nexts = set()
                previous_layers = reduce(lambda x, y: x.union(y), list(vertexes.values())[: layer]) if layer > 0 else []
                for vertex in vertexes[layer]:
                    if vertex not in previous_layers:
                        for next_vertex in vertex.connections.keys():
                            nexts.add(next_vertex)

                layer += 1
                if nexts != set():
                    vertexes[layer] = nexts
                else:
                    break

            for length in range(1, len(vertexes)):
                if node in vertexes[length]:
                    chains = np.array([[0., node]])
                    for layers in reversed(range(length)):
                        chain = copy(chains)
                        for nodes in vertexes[layers]:
                            for i in range(len(chain[:, -1])):
                                if chain[i][-1] in nodes.connections.keys():
                                    new_chain = np.append(chain[i], nodes)
                                    new_chain[0] = chains[i][0] + nodes.connections[chain[i][-1]]

                                    if len(chain[0]) == len(chains[0]):
                                        chains = np.hstack((chains, [[0]] * chains.shape[0]))
                                    chains = np.vstack((chains, new_chain))
                    for arr in chains:
                        temp = arr[1:][arr[1:] != 0]
                        if temp[0] == temp[-1] and len(temp) > 1:
                            cycles.add((arr[0], tuple(temp)))
                    break
        if verbose:
            for cycle in cycles:
                if cycle[0] > 0:
                    print('\nWeight = ', cycle[0])
                    print(list(map(lambda val: str(val), cycle[1])))
        return cycles

    def get_eigens(self):
        return np.linalg.eig(self.A)[0]

    def stability(self):
        max_eigen = max(abs(self.get_eigens()))
        if max_eigen < 1:
            return True
        else:
            return False


class UI(QDialog):
    def __init__(self):
        super(UI, self).__init__()

        # top_box
        self.top_box = QHBoxLayout()

        self.useStylePaletteCheckBox = QCheckBox("Light")
        self.useStylePaletteCheckBox.setChecked(False)

        self.reset = QPushButton("Reset")
        self.reset.setFlat(True)

        self.run = QPushButton("Execute")
        self.run.setFlat(True)

        self.top_box.addWidget(self.useStylePaletteCheckBox)
        self.top_box.addStretch(1)
        self.top_box.addWidget(self.run)
        self.top_box.addWidget(self.reset)

        # middle_box
        self.middle_box = QTabWidget()

        self.addVertex = QPushButton("Add vertex")
        self.addVertex.setFlat(True)

        self.addVertex_value = QLineEdit()
        self.addVertex_value.setPlaceholderText("Name new vertex")

        self.removeVertex = QPushButton("Remove vertex")
        self.removeVertex.setFlat(True)

        self.removeVertex_value = QLineEdit()
        self.removeVertex_value.setPlaceholderText("Name vertex to remove")

        self.addConnection = QPushButton("Add connection")
        self.addConnection.setFlat(True)

        self.addConnection_value = QLineEdit()
        self.addConnection_value.setPlaceholderText("Name vertexes to connect")

        self.removeConnection = QPushButton("Remove connection")
        self.removeConnection.setFlat(True)

        self.removeConnection_value = QLineEdit()
        self.removeConnection_value.setPlaceholderText("Name vertexes to disconnect")

        layout = QGridLayout()

        layout.addWidget(self.addVertex_value, 0, 0)
        layout.addWidget(self.addVertex, 1, 0)
        layout.addWidget(self.addConnection_value, 2, 0)
        layout.addWidget(self.addConnection, 3, 0)
        layout.addWidget(self.removeVertex_value, 4, 0)
        layout.addWidget(self.removeVertex, 5, 0)
        layout.addWidget(self.removeConnection_value, 6, 0)
        layout.addWidget(self.removeConnection, 7, 0)

        self.middle_box.setLayout(layout)

        # graph_box
        self.plot_widget = QWebEngineView()
        self.plot_widget.setHtml("<!DOCTYPE html><html><body style='background-color:grey;'></body></html>")
        self.plot_widget.setFixedHeight(500)

        # bottom_box
        self.bottom_box = QTabWidget()
        self.structural_stability = QPushButton("Check structural stability")
        self.structural_stability.setFlat(True)

        self.structural_stability_value = QTextEdit()
        self.structural_stability_value.setPlaceholderText("Here will be displayed the list of cycles")

        self.numerical_stability = QPushButton("Check numerical stability")
        self.numerical_stability.setFlat(True)

        self.numerical_stability_value = QLineEdit()
        self.numerical_stability_value.setPlaceholderText("Here will be displayed if "
                                                          "the graph is numerically stable or not")
        layout = QGridLayout()

        layout.addWidget(self.structural_stability, 0, 0)
        layout.addWidget(self.structural_stability_value, 0, 1)
        layout.addWidget(self.numerical_stability, 1, 0)
        layout.addWidget(self.numerical_stability_value, 1, 1)

        self.bottom_box.setLayout(layout)

        # widow_init
        self.setWindowIcon(QIcon('icon.jpg'))
        self.setWindowTitle("Solver")
        self.setWindowIconText('Solver')

        self.mainLayout = QGridLayout()
        self.mainLayout.addLayout(self.top_box, 0, 0, 1, 7)
        self.mainLayout.addWidget(self.middle_box, 1, 5, 4, 2)
        self.mainLayout.addWidget(self.plot_widget, 1, 0, 4, 5)
        self.mainLayout.addWidget(self.bottom_box, 6, 0, 2, 7)

        self.setLayout(self.mainLayout)

        self.resize(1100, 700)
        self.originalPalette = QApplication.palette()
        self.change_palette()

        # set_connections
        self.useStylePaletteCheckBox.toggled.connect(self.change_palette)
        self.reset.clicked.connect(self.clr)
        self.run.clicked.connect(self.execute)
        self.addVertex.clicked.connect(self.add_vertex)
        self.removeVertex.clicked.connect(self.remove_vertex)
        self.addConnection.clicked.connect(self.add_connection)
        self.removeConnection.clicked.connect(self.remove_connection)
        self.structural_stability.clicked.connect(self.check_structural_stability)
        self.numerical_stability.clicked.connect(self.check_numerical_stability)

        # graph_init
        self.graph = Graph("Graph")

    def change_palette(self):
        dark_palette = QPalette()

        QApplication.setStyle(QStyleFactory.create("Fusion"))

        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
        dark_palette.setColor(QPalette.Base, QColor(42, 42, 42))
        dark_palette.setColor(QPalette.AlternateBase, QColor(66, 66, 66))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
        dark_palette.setColor(QPalette.Dark, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.Shadow, QColor(20, 20, 20))
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
        dark_palette.setColor(QPalette.HighlightedText, Qt.white)
        dark_palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(127, 127, 127))

        if self.useStylePaletteCheckBox.isChecked():
            self.plot_graph("<!DOCTYPE html><html><body style='background-color:white;'></body></html>")
            QApplication.setPalette(QApplication.style().standardPalette())

        else:
            self.plot_graph("<!DOCTYPE html><html><body style='background-color:grey;'></body></html>")
            QApplication.setPalette(dark_palette)

    def plot_graph(self, html):
        self.plot_widget = show_qt(html)
        self.mainLayout.addWidget(self.plot_widget, 1, 0, 4, 5)
        self.plot_widget.setFixedHeight(500)

    def add_vertex(self):
        params = self.addVertex_value.text().split()
        if len(params) != 2:
            self.addVertex_value.clear()
        else:
            self.graph.add_node(Node(*params))
            self.addVertex_value.clear()

    def add_connection(self):
        pass

    def remove_vertex(self):
        params = self.removeVertex_value.text().split()
        if len(params) != 2:
            self.removeVertex_value.clear()
        else:
            self.graph.remove_node(Node(*params))
            self.removeVertex_value.clear()

    def remove_connection(self):
        params = self.removeConnection_value.text().split()
        if len(params) != 2:
            self.removeConnection_value.clear()
        else:
            # self.graph.nodesremoveConnection_value(Node(*params))
            self.removeConnection_value.clear()

    def check_structural_stability(self):
        pass

    def check_numerical_stability(self):
        pass

    def clr(self):
        pass

    def execute(self):
        self.clr()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = UI()
    main_window.show()
    app.exec_()
