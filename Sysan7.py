import networkx as nx
import plotly
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from functools import reduce
from copy import copy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QLineEdit, QDialog, \
    QHBoxLayout, QGridLayout, QStyleFactory, QCheckBox, QPushButton, QTabWidget, QTextEdit, QComboBox
from PyQt5.QtGui import QPalette, QColor, QIcon
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt
import sys


# create_swot_table.py module

class SelfDrivingCarMap:
    def __init__(self):
        actions = [
            "Invest into ML Technologies",
            "Invest into Eco-Friendly Technologies",
            "Invest into Car Technologies",
            "Invest into Crypto Technologies",
            "Invest into Software Development",
            "Invest into Advertisments",
            "Invest into Education",
            "Create Service Centres",
            "Decrease Fuel Price",
            "Use Eco-Friendly Materials",
            "Use Cheap Materials",
            "Get Government Support",
            "Boosting Technological Progress",
            "Encourage Existing Industries",
            "Boost Infrastructure",
            "Develop New Laws",
            "Develop GPS Technologies",
            "Develop Safety Systems",
            "Develop New Laws",
            "Develop Better GPS",
            "Increase Social Awareness",
        ]
        self.actions = actions
        self.connections = None

    def form_connections(self):
        self.connections = pd.read_csv("final_cognitive.csv").set_index("Unnamed: 0")
        return self.connections


# networkxplotter.py module
class NetworkXPlotter(object):
    """Class for casting graph to NetworkX graph and plotting it."""
    def __init__(self, custom_g, layout="circular", dim=2):
        """
        Args:
            custom_g(Graph): object of our own class with nodes and connections.
            layout(str): type of layout for ploting ('spring', 'circular', 'spectral', 'random').
        Raises:
            ValueError: not implemented type of layout.
        """
        # currently only dim = 2 is possible
        self.dim = dim
        self.html = None
        self.G = nx.DiGraph()
        edges = []
        weights = []
        nodes = []
        values = []
        for node in custom_g.nodes:
            nodes.append(str(node))
            values.append(node.value)
            for k, v in node.connections.items():
                if v != 0:
                    edges.append((str(node), str(k)))
                    weights.append(v)

        self.G.add_nodes_from(nodes)
        self.G.add_edges_from(edges)

        if layout == "spring":
            pos = nx.layout.spring_layout(self.G, dim=2)
        elif layout == "circular":
            pos = nx.layout.circular_layout(self.G, dim=2)
        elif layout == "spectral":
            pos = nx.layout.spectral_layout(self.G, dim=2)
        elif layout == "random":
            pos = nx.layout.random_layout(self.G, dim=2)
        else:
            raise ValueError(
                "No such layout. Try one of the following: 'spring', 'circular', 'spectral', 'random'."
            )

        index = 0
        for node in self.G.nodes:
            self.G.nodes[node]["pos"] = list(pos[node])
            self.G.nodes[node]["name"] = str(node)
            self.G.nodes[node]["value"] = values[index]
            index += 1

        index = 0
        for edge in self.G.edges:
            self.G.edges[edge]["start"] = str(edge[0])
            self.G.edges[edge]["end"] = str(edge[1])
            self.G.edges[edge]["weight"] = weights[index]
            index += 1

    def plot(
        self,
        colorscale="Inferno",
        reversescale=False,
        edge_opacity=0.5,
        edge_color="Black",
        node_opacity=0.5,
        node_size=10,
        edge_scale=0,
        title="Network Graph",
        fontsize=16,
        paper_bgcolor=None,
        plot_bgcolor=None,
        plot_text=False,
        text_color=None,
    ):
        """
        Args:edge_opacity
            save_html(bool): True to save html code as a string
            colorscale(str): Plotly colorscale for nodes (see below). Mostly the same as matplotlib colormaps
            reversescale(bool): True to use reveresed colorscale. use 'colorscalename_r' in colorscale to also do it
            edge_opacity(float): opacity of edges from 0 to 1
            edge_color(str): CSS string name of color for edges
            node_opacity(float): opacity of nodes from 0 to 1
            node_size(int): size of nodes without self-connection, others are x1.5 bigger
            edge_scale(float): multiplier of edge size. Formula edge_width = edge_scale*edge_weight + 1
            title(str): title of the plot
            fontsize(int): fontsize of the title
            paper_bgcolor(str or None): (Experiment) change background color None or
            'rgba(xxx,xxx,xxx,0.3)' - not recommended
            plot_bgcolor(str or None): (Experiment) change background color None or
            'rgba(xxx,xxx,xxx,0.5)' - not recommended
            plot_text(bool): True to write names of nodes on figure
            text_color(str or None): text color
        Returns:
            self(NetworkXPlotter) object of NetworkXPlotter class
        """
        # colorscale options
        # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' | 'Magma'
        # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis'| 'Inferno'
        # look up for Plotly colorscale in internet for all options
        trace_recode = []

        weights = []
        for edge in self.G.edges:
            weights.append(self.G.edges[edge]["weight"])

        weights = np.array(weights) / np.max(np.abs(weights))
        index = 0
        for edge in self.G.edges:
            x0, y0 = self.G.nodes[edge[0]]["pos"]
            x1, y1 = self.G.nodes[edge[1]]["pos"]
            weight = edge_scale * abs(weights[index]) + 1
            trace = go.Scatter(
                x=tuple([x0, x1, None]),
                y=tuple([y0, y1, None]),
                mode="lines",
                line={"width": weight, "color": edge_color},
                line_shape="spline",
                opacity=edge_opacity,
            )
            trace_recode.append(trace)
            index += 1

        middle_hover_trace = go.Scatter(
            x=[],
            y=[],
            hovertext=[],
            mode="markers",
            hoverinfo="text",
            marker={"size": 20, "color": edge_color},
            opacity=0,
        )

        for edge in self.G.edges:
            x0, y0 = self.G.nodes[edge[0]]["pos"]
            x1, y1 = self.G.nodes[edge[1]]["pos"]
            hovertext = (
                "From: "
                + str(self.G.edges[edge]["start"])
                + "<br>"
                + "To: "
                + str(self.G.edges[edge]["end"])
                + "<br>"
                + "Weight: "
                + str(self.G.edges[edge]["weight"])
            )
            middle_hover_trace["x"] += tuple([(x0 + x1) / 2])
            middle_hover_trace["y"] += tuple([(y0 + y1) / 2])
            middle_hover_trace["hovertext"] += tuple([hovertext])

        trace_recode.append(middle_hover_trace)

        node_trace = go.Scatter(
            x=[],
            y=[],
            hovertext=[],
            text=[],
            mode="markers+text" if plot_text else "markers",
            textposition="bottom center",
            hoverinfo="text",
            marker=dict(
                line=dict(width=node_size / 10, color="Black"),
                showscale=True,
                colorscale=colorscale,
                reversescale=reversescale,
                opacity=node_opacity,
                color=[],
                size=[],
                colorbar=dict(
                    thickness=10,
                    title="Node Connections",
                    xanchor="left",
                    titleside="right",
                    titlefont=dict(size=fontsize, color=text_color),
                    tickfont=dict(size=fontsize, color=text_color),
                ),
            ),
            textfont=dict(color=text_color),
        )

        for node in self.G.nodes():
            x, y = self.G.nodes[node]["pos"]
            text = str(self.G.nodes[node]["name"])
            node_trace["x"] += tuple([x])
            node_trace["y"] += tuple([y])
            node_trace["text"] += tuple([text])

        for node, adjacencies in self.G.adjacency():
            if node in adjacencies.keys():
                node_trace["marker"]["size"] += tuple([1.5 * node_size])
            else:
                node_trace["marker"]["size"] += tuple([node_size])
            node_trace["marker"]["color"] += tuple([len(adjacencies)])
            text = "Name: " + str(self.G.nodes[node]["name"])
            node_info = (
                text
                + "<br>Value: "
                + str(self.G.nodes[node]["value"])
                + "<br># of connections: "
                + str(len(adjacencies))
            )
            node_trace["hovertext"] += tuple([node_info])

        trace_recode.append(node_trace)

        fig = go.Figure(
            data=trace_recode,
            layout=go.Layout(
                title=title,
                showlegend=False,
                hovermode="closest",
                titlefont=dict(size=fontsize, color=text_color),
                margin={"b": 40, "l": 40, "r": 40, "t": 40},
                xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                autosize=True,
                paper_bgcolor=paper_bgcolor,
                plot_bgcolor=plot_bgcolor,
                annotations=[
                    dict(
                        ax=(
                            self.G.nodes[edge[0]]["pos"][0]
                            + self.G.nodes[edge[1]]["pos"][0]
                        )
                        / 2,
                        ay=(
                            self.G.nodes[edge[0]]["pos"][1]
                            + self.G.nodes[edge[1]]["pos"][1]
                        )
                        / 2,
                        axref="x",
                        ayref="y",
                        x=(
                            self.G.nodes[edge[1]]["pos"][0] * 3
                            + self.G.nodes[edge[0]]["pos"][0]
                        )
                        / 4,
                        y=(
                            self.G.nodes[edge[1]]["pos"][1] * 3
                            + self.G.nodes[edge[0]]["pos"][1]
                        )
                        / 4,
                        xref="x",
                        yref="y",
                        showarrow=True,
                        arrowcolor=edge_color,
                        arrowhead=4,
                        arrowsize=2,
                        arrowwidth=edge_scale * 0.05 + 1,
                        opacity=edge_opacity,
                    )
                    for edge in self.G.edges
                ],
            ),
        )
        self.html = "<html><body>"
        self.html += plotly.offline.plot(
            fig, output_type="div", include_plotlyjs="cdn",
        )
        self.html += "</body></html>"
        return self


# Sysan7.py module
def show_qt(raw_html):
    fig_view = QWebEngineView()
    fig_view.setHtml(raw_html)
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


def form_sw_table(swot, t_count, s_count=12):
    sw = swot.index
    sw_table = np.zeros((len(sw), len(sw)))

    for i in range(len(sw) - 1):
        for j in range(i + 1, len(sw)):
            weight = 0
            if (i < s_count and j < s_count) or (i >= s_count and j >= s_count):
                for k in range(len(swot.iloc[i])):
                    weight = weight - min(swot.iloc[i][k], swot.iloc[j][k]) if k < t_count \
                        else weight + min(swot.iloc[i][k], swot.iloc[j][k])

            sw_table[i, j] = round(weight, 5)

    sw_table = sw_table + sw_table.T
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
                        if node_right in clusters_values[j] and node_right not in clusters_values[i] else 0
                    weight += additive
            connections.append([weight, subgraphs[j]])

        subgraphs[i].set_connections(connections)

    aggregated = Graph("aggregated")
    aggregated.set_nodes(subgraphs)
    return aggregated


class Graph(object):
    def __init__(self, name):
        # print("Let me create that starnge GRAPH boi!")
        self.name = name
        self.nodes = []
        self.A = []

    def from_pandas(self, table):
        data = table.form_connections()
        nodes = dict()

        for index in data.index:
            if index not in table.actions:
                nodes[index] = Node(index, 0)
            else:
                nodes[index] = Node(index, 1)

        for index in data.index:
            for column in data.columns:
                if data.loc[index][column] != 0:
                    nodes[index].set_weight([data.loc[index][column], nodes[column]])
        self.set_nodes(list(nodes.values()))

    def clear(self):
        self.nodes = []
        self.A = []

    def set_nodes(self, nodes):
        self.nodes = nodes

    def get_node(self, name):
        try:
            for node in self.nodes:
                if name == node.name:
                    return node
            raise ValueError()
        except ValueError:
            print("No such node")

    def add_node(self, node):
        self.nodes.append(node)

    def remove_node(self, node):
        try:
            self.nodes.remove(node)
            for vertex in self.nodes:
                vertex.remove_connection(node)
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

    def send_impulses(self, impulse, duration, overall_duration_multiplier=5):
        previous = np.zeros((len(self.nodes), 1))
        current = np.array([[node.value] for node in self.nodes])
        graphical = {node.name: [] for node in self.nodes}

        for i in range(impulse.shape[0]):
            self.nodes[i].set_value(1) if impulse[i] != 0 else 0

        for i in range(duration * overall_duration_multiplier):

            impulse = impulse if i < duration * overall_duration_multiplier else np.zeros((impulse.shape[0], 1))

            new = current + np.array(self.A).T @ (current - previous) + impulse

            nodes_values = {self.nodes[i]: current[i, 0] / np.linalg.norm(current, np.inf)
                            for i in range(len(self.nodes))}

            previous = current
            current = new
            normalizer = sum(list(map(lambda node: node.value, self.nodes[-5:])))

            for j in range(1, 6):
                self.nodes[-j].set_value(self.nodes[-j].value / normalizer)

            for index in range(len(self.nodes)):
                graphical[self.nodes[index].name].append(nodes_values[self.nodes[index]])

            self.update_values(nodes_values)

        return graphical

    def search_cycles(self, limit=3, verbose=False):
        cycles = set()
        for node in self.nodes:
            layer = 0
            vertexes = dict()
            vertexes[layer] = {node}
            while layer < min(limit, len(self.nodes)):
                nexts = set()
                for vertex in vertexes[layer]:
                    for next_vertex in vertex.connections.keys():
                        nexts.add(next_vertex)

                layer += 1
                if nexts != set():
                    vertexes[layer] = nexts
                else:
                    break

            for length in range(1, len(vertexes)):
                if node in vertexes[length]:
                    vertexes[length].remove(node)
                    chains = np.array([[0., node]])
                    for layers in reversed(range(length)):
                        chain = copy(chains)
                        for nodes in vertexes[layers]:
                            for i in range(len(chain[:, -1])):
                                if chain[i][-1] in nodes.connections.keys() and chain[i][-1] != nodes:
                                    new_chain = np.append(chain[i], nodes)
                                    new_chain[0] = chains[i][0] * np.sign(nodes.connections[chain[i][-1]])

                                    if len(chain[0]) == len(chains[0]):
                                        chains = np.hstack((chains, [[0]] * chains.shape[0]))
                                    chains = np.vstack((chains, new_chain))
                    for arr in chains:
                        temp = arr[1:][arr[1:] != 0]
                        if temp[0] == temp[-1] and len(temp) > 1:
                            cycles.add((np.round(arr[0], 5), tuple(temp[::-1])))

        if verbose:
            for cycle in cycles:
                print("\nWeight = ", cycle[0])
                print(list(map(lambda val: str(val), cycle[1])))

        unstable_cycles = list(filter(lambda el: el[0] > 0, cycles))
        return unstable_cycles

    def get_eigens(self):
        return np.linalg.eig(self.A)[0]

    def stability(self):
        try:
            max_eigen = max(abs(self.get_eigens()))
            if max_eigen < 1:
                return True
            else:
                return False
        except np.linalg.LinAlgError:
            print("one dimensional array")
            return False


class Canvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, title=None):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.set_title(title)
        self.compute_initial_figure()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

    def compute_initial_figure(self):
        pass


class UI(QDialog):
    def __init__(self):
        super(UI, self).__init__()
        # top_box
        self.top_box = QHBoxLayout()

        self.switch_color_mode_button = QCheckBox("Light")
        self.switch_color_mode_button.setChecked(False)

        self.plot_scenario_dynamics_button = QPushButton("Plot scenario dynamics")
        self.plot_scenario_dynamics_button.setFlat(True)

        self.send_impulse_button = QPushButton("Send impulse")
        self.send_impulse_button.setFlat(True)

        self.load_cognitive_map_button = QPushButton("Load cognitive map")
        self.load_cognitive_map_button.setFlat(True)

        self.generate_random_graph_button = QPushButton("Generate random graph")
        self.generate_random_graph_button.setFlat(True)

        self.reset_outputs_button = QPushButton("Reset outputs")
        self.reset_outputs_button.setFlat(True)

        self.top_box.addWidget(self.switch_color_mode_button)
        self.top_box.addStretch(1)
        self.top_box.addWidget(self.plot_scenario_dynamics_button)
        self.top_box.addWidget(self.send_impulse_button)
        self.top_box.addWidget(self.load_cognitive_map_button)
        self.top_box.addWidget(self.generate_random_graph_button)
        self.top_box.addWidget(self.reset_outputs_button)

        # middle_box
        self.middle_box = QTabWidget()

        self.add_vertex_button = QPushButton("Add vertex")
        self.add_vertex_button.setFlat(True)

        self.add_vertex_value = QLineEdit()
        self.add_vertex_value.setPlaceholderText("Name new vertex")

        self.remove_vertex_button = QPushButton("Remove vertex")
        self.remove_vertex_button.setFlat(True)

        self.remove_vertex_value = QLineEdit()
        self.remove_vertex_value.setPlaceholderText("Name vertex to remove")

        self.add_connection_button = QPushButton("Add connection")
        self.add_connection_button.setFlat(True)

        self.add_connection_value = QLineEdit()
        self.add_connection_value.setPlaceholderText("Name vertexes to connect")

        self.remove_connection_button = QPushButton("Remove connection")
        self.remove_connection_button.setFlat(True)

        self.alter_connection_button = QPushButton("Alter connection")
        self.alter_connection_button.setFlat(True)

        self.alter_connection_value = QLineEdit()
        self.alter_connection_value.setPlaceholderText("Name vertexes to alter connection")

        self.remove_connection_value = QLineEdit()
        self.remove_connection_value.setPlaceholderText("Name vertexes to disconnect")

        layout = QGridLayout()

        layout.addWidget(self.add_vertex_button, 0, 0)
        layout.addWidget(self.add_vertex_value, 1, 0)
        layout.addWidget(self.add_connection_button, 2, 0)
        layout.addWidget(self.add_connection_value, 3, 0)
        layout.addWidget(self.remove_vertex_button, 4, 0)
        layout.addWidget(self.remove_vertex_value, 5, 0)
        layout.addWidget(self.remove_connection_button, 6, 0)
        layout.addWidget(self.remove_connection_value, 7, 0)
        layout.addWidget(self.alter_connection_button, 8, 0)
        layout.addWidget(self.alter_connection_value, 9, 0)

        self.middle_box.setLayout(layout)

        # graph_box
        self.plot_widget = QWebEngineView()
        self.plot_widget.setHtml("<!DOCTYPE html><html><body style='background-color:grey;'></body></html>")
        self.plot_widget.setFixedHeight(500)

        # bottom_box
        self.bottom_box = QTabWidget()
        self.check_structural_stability_button = QPushButton("Check structural stability")
        self.check_structural_stability_button.setFlat(True)

        self.structural_stability_value = QTextEdit()
        self.structural_stability_value.setPlaceholderText("Here will be displayed the list of cycles")

        self.check_numerical_stability_button = QPushButton("Check numerical stability")
        self.check_numerical_stability_button.setFlat(True)

        self.numerical_stability_value = QLineEdit()
        self.numerical_stability_value.setPlaceholderText("Here will be displayed if "
                                                          "the graph is numerically stable or not")
        layout = QGridLayout()

        layout.addWidget(self.check_structural_stability_button, 0, 0)
        layout.addWidget(self.structural_stability_value, 0, 1)
        layout.addWidget(self.check_numerical_stability_button, 1, 0)
        layout.addWidget(self.numerical_stability_value, 1, 1)

        self.bottom_box.setLayout(layout)

        # graph_init
        self.graph = Graph("Graph")
        self.netplot = None
        self.node_dynamics_dict = None

        # widow_init
        self.setWindowIcon(QIcon("icon.jpg"))
        self.setWindowTitle("Cognitive Model")
        self.setWindowIconText("Cognitive Model")

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
        self.switch_color_mode_button.toggled.connect(self.change_palette)
        self.send_impulse_button.clicked.connect(self.send_impulse)
        self.generate_random_graph_button.clicked.connect(self.generate_random_graph)
        self.add_vertex_button.clicked.connect(self.add_vertex)
        self.remove_vertex_button.clicked.connect(self.remove_vertex)
        self.add_connection_button.clicked.connect(self.add_connection)
        self.remove_connection_button.clicked.connect(self.remove_connection)
        self.check_structural_stability_button.clicked.connect(self.check_structural_stability)
        self.check_numerical_stability_button.clicked.connect(self.check_numerical_stability)
        self.load_cognitive_map_button.clicked.connect(self.show_swot_graph)
        self.reset_outputs_button.clicked.connect(self.reset_outputs)
        self.alter_connection_button.clicked.connect(self.alter_connection)
        self.plot_scenario_dynamics_button.clicked.connect(self.plot_scenario_dynamics)

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

        if self.switch_color_mode_button.isChecked():
            if len(self.graph.nodes) == 0:
                self.plot_graph("<!DOCTYPE html><html><body style='background-color:white;'></body></html>")
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            if len(self.graph.nodes) == 0:
                self.plot_graph("<!DOCTYPE html><html><body style='background-color:grey;'></body></html>")
            QApplication.setPalette(dark_palette)

    def plot_graph(self, html=None):
        self.plot_widget = show_qt(self.netplot.html) if html is None else show_qt(html)
        self.mainLayout.addWidget(self.plot_widget, 1, 0, 4, 5)
        self.plot_widget.setFixedHeight(500)

    def add_vertex(self):
        params = self.add_vertex_value.text().split(' ; ')  # params[0] - name; [1] - weight;
        if len(params) != 2:
            self.add_vertex_value.clear()
        else:
            self.graph.add_node(Node(*params))
            self.add_vertex_value.clear()
            self.create_graph_html()
            self.plot_graph()

    def add_connection(self):
        params = self.add_connection_value.text().split(' ; ')  # params[0] - from; [1] - to; [2] - weight
        if len(params) != 3:
            self.add_connection_value.clear()
        else:
            node_from = self.graph.get_node(params[0])
            node_to = self.graph.get_node(params[1])
            weight = float(params[2])
            node_from.set_connections([[weight, node_to]])
            # node_to.set_connections([[weight, node_from]])  # for two-way connection
            self.add_connection_value.clear()
            self.create_graph_html()
            self.plot_graph()

    def remove_vertex(self):
        params = self.remove_vertex_value.text().split(' ; ')
        if len(params) != 1:
            self.remove_vertex_value.clear()
        else:
            self.graph.remove_node(self.graph.get_node(params[0]))
            self.remove_vertex_value.clear()
            self.create_graph_html()
            self.plot_graph()

    def remove_connection(self):
        params = self.remove_connection_value.text().split(' ; ')  # params[0] - from; [1] - to;
        if len(params) != 2:
            self.remove_connection_value.clear()
        else:
            node_from = self.graph.get_node(params[0])
            node_to = self.graph.get_node(params[1])
            node_from.remove_connection(node_to)
            # node2.remove_connection(node1)  # for two-way connection
            self.remove_connection_value.clear()
            self.create_graph_html()
            self.plot_graph()

    def alter_connection(self):
        params = self.alter_connection_value.text().split(' ; ')  # params[0] - node1; [1] - node2; [2] - weight;
        if len(params) != 3:
            self.alter_connection_value.clear()
        else:
            node_from = self.graph.get_node(params[0])
            node_to = self.graph.get_node(params[1])
            weight = float(params[2])
            node_from.set_weight([weight, node_to])
            # node_to.set_weight([weight, node_from])  # for two-way connection
            self.alter_connection_value.clear()
            self.create_graph_html()
            self.plot_graph()

    def check_structural_stability(self):
        try:
            cycles = self.graph.search_cycles()
            cycles = reduce(lambda a, b: a + b, ["Weight: " + str(np.round(np.array(cycle[0]), 2)) + "  |  Cycle: "
                                                 + reduce(lambda x, y: x + y, [str(node) +
                                                                               " -> " for node in cycle[1]])[:-4] + "\n"
                                                 for cycle in cycles])

            self.structural_stability_value.setText(str(cycles))
        except TypeError:
            self.structural_stability_value.setText("No cycles detected")

    def check_numerical_stability(self):
        stability = "Stable" if self.graph.stability() else "Not Stable"
        self.numerical_stability_value.setText(stability)
        pass

    def generate_random_graph(self):
        likelyhood = 0.4
        nods = [
                   Node(letter, number)
                   for letter, number in zip(list(map(lambda x: str(x), range(44))), range(44))][:8]
        for nod in nods:
            nod.set_connections(
                [
                    [rand, x]
                    for rand, x in zip(np.random.uniform(0, 1, len(nods)), nods)
                    if np.random.binomial(1, likelyhood)
                ]
            )
        self.graph = Graph("Graph")
        self.graph.set_nodes(nods)
        self.create_graph_html()
        self.plot_graph()

    def show_swot_graph(self):
        table = SelfDrivingCarMap()
        self.graph = Graph("Cognitive map")
        self.graph.from_pandas(table)
        self.create_graph_html()
        self.plot_graph()

    def create_graph_html(self):
        self.netplot = NetworkXPlotter(self.graph, layout='circular')
        if self.switch_color_mode_button.isChecked():
            self.netplot.plot(
                colorscale="sunset",
                edge_opacity=0.6,
                edge_color="SlateGrey",
                node_opacity=1,
                node_size=12,
                edge_scale=3,
                title="Cognitive network visualization<br>(wider edges have higher weights, "
                      "bigger nodes have self edge)",
                fontsize=12,
                plot_text=True
            )
        else:
            self.netplot.plot(
                colorscale="sunset",
                edge_opacity=0.6,
                edge_color="WhiteSmoke",
                node_opacity=1,
                node_size=12,
                edge_scale=3,
                title="Cognitive network visualization (Dark)<br>(wider edges have higher weights, "
                      "bigger nodes have self edge)",
                fontsize=12,
                plot_text=True,
                paper_bgcolor='rgba(42,42,42,1)',
                plot_bgcolor='rgba(53,53,53,1)',
                text_color="Silver"
            )

    def send_impulse(self):
        dialog = QDialog(self)
        data_for_calculations = {node.name: 0 for node in self.graph.nodes}
        dialog.setWindowIcon(QIcon("icon.jpg"))
        dialog.setWindowTitle("Impulse settings")
        dialog.setWindowIconText("Impulse settings")

        dialog.mainLayout = QGridLayout()

        dialog.select_nodes_combobox = QComboBox()
        dialog.select_nodes_combobox.addItems([node.name for node in self.graph.nodes])

        dialog.select_impulse_magnitude = QLineEdit("")
        dialog.select_impulse_magnitude.setPlaceholderText("Enter impulse magnitude")
        dialog.select_impulse_magnitude.setFixedWidth(200)
        dialog.text_box = QTextEdit("")
        dialog.text_box.setPlaceholderText("Here will be displayed a list of added nodes and their weights")

        dialog.impulse_duration = QLineEdit("")
        dialog.impulse_duration.setPlaceholderText("Enter impulse duration")

        dialog.append_impulse = QPushButton("Append impulse")
        dialog.append_impulse.setFlat(True)

        dialog.send_impulse = QPushButton("Form impulse")
        dialog.send_impulse.setFlat(True)

        dialog.mainLayout.addWidget(dialog.select_nodes_combobox, 0, 0, 1, 3)
        dialog.mainLayout.addWidget(dialog.select_impulse_magnitude, 0, 3, 1, 1)
        dialog.mainLayout.addWidget(dialog.append_impulse, 0, 4, 1, 1)
        dialog.mainLayout.addWidget(dialog.text_box, 1, 0, 1, 5)
        dialog.mainLayout.addWidget(dialog.impulse_duration, 2, 0, 1, 4)
        dialog.mainLayout.addWidget(dialog.send_impulse, 2, 4, 1, 1)

        def send_data():
            self.graph.form_connection_matrix()
            self.node_dynamics_dict = self.graph.send_impulses(np.array(list(map(lambda x: float(x),
                                                                                 data_for_calculations.values()))),
                                                               int(dialog.impulse_duration.text()))
            # self.graphical = np.array([dynamic for dynamic in self.graphical.values()]).T
            self.create_graph_html()
            self.plot_graph()
            dialog.close()

        def append_data():
            dialog.text_box.append(str(dialog.select_nodes_combobox.currentText() +
                                       " | Magnitude: " + dialog.select_impulse_magnitude.text()))
            data_for_calculations[dialog.select_nodes_combobox.currentText()] = dialog.select_impulse_magnitude.text()

            dialog.select_impulse_magnitude.clear()

        dialog.send_impulse.clicked.connect(send_data)
        dialog.append_impulse.clicked.connect(append_data)

        dialog.setLayout(dialog.mainLayout)
        dialog.resize(500, 200)

        dialog.show()

    def reset_outputs(self):
        self.graph.clear()
        self.netplot = None
        self.structural_stability_value.clear()
        self.numerical_stability_value.clear()
        self.add_vertex_value.clear()
        self.add_connection_value.clear()
        self.remove_vertex_value.clear()
        self.remove_connection_value.clear()
        self.alter_connection_value.clear()
        self.plot_widget = QWebEngineView()
        if self.switch_color_mode_button.isChecked():
            self.plot_graph("<!DOCTYPE html><html><body style='background-color:white;'></body></html>")
        else:
            self.plot_graph("<!DOCTYPE html><html><body style='background-color:grey;'></body></html>")
        self.plot_widget.setFixedHeight(500)

    def plot_scenario_dynamics(self):
        dialog = QDialog(self)
        data_for_plot = {node.name: False for node in self.graph.nodes}
        dialog.setWindowIcon(QIcon("icon.jpg"))
        dialog.setWindowTitle("Scenario dynamics")
        dialog.setWindowIconText("Scenario dynamics")

        dialog.mainLayout = QGridLayout()

        dialog.select_nodes_combobox = QComboBox()
        dialog.select_nodes_combobox.addItems([node.name for node in self.graph.nodes])

        dialog.text_box = QTextEdit("")
        dialog.text_box.setPlaceholderText("Here will be displayed a list of added vertexes")

        dialog.append_vertex_button = QPushButton("Append vertex")
        dialog.append_vertex_button.setFlat(True)

        dialog.plot_dynamics_button = QPushButton("Plot")
        dialog.plot_dynamics_button.setFlat(True)

        dialog.mainLayout.addWidget(dialog.select_nodes_combobox, 0, 0, 1, 3)
        dialog.mainLayout.addWidget(dialog.append_vertex_button, 0, 3, 1, 1)
        dialog.mainLayout.addWidget(dialog.text_box, 1, 0, 1, 5)
        dialog.mainLayout.addWidget(dialog.plot_dynamics_button, 0, 4, 1, 1)

        def plot_dynamics():
            plot_dialog = QDialog(dialog)
            plot_dialog.canvas = Canvas(title="Scenario dynamics")
            plot_keys = list(filter(lambda name: data_for_plot[name], self.node_dynamics_dict.keys()))
            plot_values = np.array([self.node_dynamics_dict[key] for key in plot_keys]).T
            plot_dialog.canvas.axes.plot(np.arange(0, len(plot_values)),
                                         plot_values, lw=1)
            plot_dialog.canvas.axes.legend(plot_keys)
            plot_dialog.canvas.draw()

            layout = QGridLayout()
            layout.addWidget(plot_dialog.canvas)

            plot_dialog.setLayout(layout)
            plot_dialog.resize(800, 500)
            plot_dialog.show()
            dialog.close()

        def append_data():
            dialog.text_box.append(str(dialog.select_nodes_combobox.currentText()))
            data_for_plot[dialog.select_nodes_combobox.currentText()] = True

        dialog.plot_dynamics_button.clicked.connect(plot_dynamics)
        dialog.append_vertex_button.clicked.connect(append_data)

        dialog.setLayout(dialog.mainLayout)
        dialog.resize(500, 200)

        dialog.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = UI()
    main_window.show()
    app.exec_()
