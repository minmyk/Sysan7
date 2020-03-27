import networkx as nx
import plotly
import plotly.graph_objs as go


class NetworkXPlotter(object):
    """Class for casting graph to NetworkX graph and plotting it."""

    def __init__(self, custom_g, layout="spring", dim=2):
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
        for node in custom_g.nodes:
            for k, v in node.connections.items():
                edges.append((str(node), str(k)))
                weights.append(v)
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

        for node in self.G.nodes:
            self.G.nodes[node]["pos"] = list(pos[node])
            self.G.nodes[node]["name"] = str(node)
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
        text_color = None
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
        for edge in self.G.edges:
            x0, y0 = self.G.nodes[edge[0]]["pos"]
            x1, y1 = self.G.nodes[edge[1]]["pos"]
            weight = edge_scale * self.G.edges[edge]["weight"] + 1
            trace = go.Scatter(
                x=tuple([x0, x1, None]),
                y=tuple([y0, y1, None]),
                mode="lines",
                line={"width": weight, "color": edge_color},
                line_shape="spline",
                opacity=edge_opacity,
            )
            trace_recode.append(trace)

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
                    titlefont=dict(size=fontsize, color = text_color),
                    tickfont=dict(size=fontsize, color = text_color),
                ),
            ),
            textfont=dict(
                color=text_color
            )
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
            node_info = text + "<br># of connections: " + str(len(adjacencies))
            node_trace["hovertext"] += tuple([node_info])

        trace_recode.append(node_trace)

        fig = go.Figure(
            data=trace_recode,
            layout=go.Layout(
                title=title,
                showlegend=False,
                hovermode="closest",
                titlefont=dict(size=fontsize, color = text_color),
                margin={"b": 40, "l": 40, "r": 40, "t": 40},
                xaxis={"showgrid": False, "zeroline": False, "showticklabels": False, },
                yaxis={"showgrid": False, "zeroline": False, "showticklabels": False, },
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

        #fig.show()
        return self
