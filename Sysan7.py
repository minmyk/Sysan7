import numpy as np
from functools import reduce
from itertools import product
from networkplotter import NetworkXPlotter
from PyQt5.QtWidgets import QTableWidget, QLabel, QApplication, QLineEdit, QDialog, QGroupBox, \
    QHBoxLayout, QGridLayout, QStyleFactory, QCheckBox, QPushButton, QWidget, QTableWidgetItem, QTabWidget, \
    QHeaderView
from PyQt5.QtGui import QPalette, QColor, QIcon
from PyQt5.QtCore import Qt
import sys


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
                    additive = node_left.connections[node_right] if node_right in clusters_values[j] \
                                                                    and not node_right in clusters_values[i] else 0
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

            for length in range(1, len(vertexes)):
                if node in vertexes[length]:
                    additive = list(product(*[vertexes[i] for i in range(0, length)]))
                    for cycle in additive:
                        cycles.add(tuple(set(cycle)))

        for cycle in cycles:
            print(list(map(lambda val: str(val), cycle)))
            print("----------")


class UI(QDialog):
    def __init__(self, parent=None):
        super(UI, self).__init__(parent)

        self.reset = QPushButton("Reset")
        self.run = QPushButton("Execute")
        self.useStylePaletteCheckBox = QCheckBox("Light")
        self.tab1hbox = QHBoxLayout()
        self.tab2hbox = QHBoxLayout()
        self.Btab1 = QWidget()
        self.Btab2 = QWidget()
        self.tables = [QTableWidget(self.Btab1), QTableWidget(self.Btab1),
                       QTableWidget(self.Btab2), QTableWidget(self.Btab2)]

        self.bottomBox = QTabWidget()

        self.Mlabel2 = QLabel("Optimal weights TOPSIS:")
        self.Mlabel1 = QLabel("Optimal weights VITOR:")
        self.MspinBox2 = QLineEdit("")
        self.MspinBox1 = QLineEdit("")
        self.middleBox = QGroupBox("Optimal strategies")

        self.outputs = []
        self.results = QLabel()
        self.originalPalette = QApplication.palette()
        self.topBox = QHBoxLayout()
        self.setWindowIcon(QIcon('icon.jpg'))
        self.setWindowTitle("SWOT")
        self.setWindowIconText('SWOT')

        self.create_top_box()
        self.create_bottom_box()
        self.create_middle_box()

        self.mainLayout = QGridLayout()
        self.mainLayout.addLayout(self.topBox, 0, 0)
        self.mainLayout.addWidget(self.middleBox, 1, 0)
        self.mainLayout.addWidget(self.bottomBox, 2, 0)
        self.setLayout(self.mainLayout)
        self.resize(800, 600)
        self.change_palette()

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
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            QApplication.setPalette(dark_palette)

    def create_top_box(self):
        self.useStylePaletteCheckBox.setChecked(False)
        self.reset.setFlat(True)
        self.run.setFlat(True)
        self.topBox.addWidget(self.useStylePaletteCheckBox)
        self.topBox.addStretch(1)
        self.topBox.addWidget(self.run)
        self.topBox.addWidget(self.reset)

    def create_bottom_box(self):
        for index in range(len(self.tables)):
            self.tables[index].setColumnCount(2)
            if index in (0, 1):
                self.tables[index].setRowCount(12)
            else:
                self.tables[index].setRowCount(10)
            header = self.tables[index].horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            header.setSectionResizeMode(1, QHeaderView.Stretch)

        self.tables[0].setHorizontalHeaderLabels(["Strength level", "F"])
        self.tables[1].setHorizontalHeaderLabels(["Weakness level", "F"])
        self.tables[2].setHorizontalHeaderLabels(["Opportunities level", "F"])
        self.tables[3].setHorizontalHeaderLabels(["Threats level", "F"])

        self.tab1hbox.setContentsMargins(5, 5, 5, 5)
        self.tab1hbox.addWidget(self.tables[0])
        self.tab1hbox.addWidget(self.tables[1])
        self.Btab1.setLayout(self.tab1hbox)

        self.tab2hbox.setContentsMargins(5, 5, 5, 5)
        self.tab2hbox.addWidget(self.tables[2])
        self.tab2hbox.addWidget(self.tables[3])
        self.Btab2.setLayout(self.tab2hbox)

        self.bottomBox.addTab(self.Btab1, "Strengths / Weaknesses")
        self.bottomBox.addTab(self.Btab2, "Opportunities / Threats")

    def create_middle_box(self):
        self.outputs.append(self.MspinBox1)
        self.outputs.append(self.MspinBox2)

        layout = QGridLayout()
        self.MspinBox1.setPlaceholderText("Here will be displayed optimal VITOR strategy "
                                          "after calculations are finished.")
        self.MspinBox2.setPlaceholderText("Here will be displayed optimal TOPSIS strategy "
                                          "after calculations are finished.")
        layout.addWidget(self.MspinBox1, 0, 1, 1, 4)
        layout.addWidget(self.MspinBox2, 1, 1, 1, 4)
        layout.addWidget(self.Mlabel1, 0, 0, 1, 1)
        layout.addWidget(self.Mlabel2, 1, 0)

        self.middleBox.setLayout(layout)

    def clr(self):
        self.MspinBox1.setText("")
        self.MspinBox2.setText("")
        for table in self.tables:
            table.clearContents()

    def fill_table(self, table_index, table_to_fill):
        for index in range(len(table_to_fill.to_numpy())):
            item1 = QTableWidgetItem(list(table_to_fill.index)[index])
            item1.setTextAlignment(Qt.AlignHCenter)
            self.tables[table_index].setItem(index, 0, item1)
            item2 = QTableWidgetItem(str(np.round(table_to_fill.to_numpy(), 3)[index]))
            item2.setTextAlignment(Qt.AlignHCenter)
            self.tables[table_index].setItem(index, 1, item2)

    def execute(self):
        self.clr()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = UI()
    main_window.show()
    main_window.useStylePaletteCheckBox.toggled.connect(main_window.change_palette)
    main_window.reset.clicked.connect(main_window.clr)
    main_window.run.clicked.connect(main_window.execute)
    app.exec_()



