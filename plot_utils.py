from matplotlib import animation, rc
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
plt.rcParams["animation.html"] = "jshtml"
import matplotlib.cbook
import networkx as nx

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


def plot_each(v, e, show_gt=False, kidnap_locs=None):
    instr_id = v['instr_id']
    gt = e.gt[int(instr_id.split('_')[0])]
    graph = e.graphs[gt['scan']]
    
    gt_path = gt['path']
    
    node_pos = nx.get_node_attributes(graph,'position')
    for k, vv in node_pos.items():
        node_pos[k] = vv[:-1]

    # Extract the x, y coordinates of nodes in the path
    rel_pos = [node_pos[vp] for vp in v['path']]
    rel_x = [r[0] for r in rel_pos]
    rel_y = [r[1] for r in rel_pos]

    gt_rel_pos = [node_pos[vp] for vp in gt_path]
    gt_rel_x = [r[0] for r in gt_rel_pos]
    gt_rel_y = [r[1] for r in gt_rel_pos]
    
    rel_x.extend(gt_rel_x)
    rel_y.extend(gt_rel_y)
    
    xlim = [min(rel_x)-3, max(rel_x)+3]
    ylim = [min(rel_y)-3, max(rel_y)+3]
    
    _G = nx.DiGraph(graph)
    _edges = []
    _gt_edges = []
    last_vp = None
    fig, ax = plt.subplots()
    for idx, vp in enumerate(v['path']):
        if last_vp:
            _G.add_edge(last_vp, vp)
            _edges.append((last_vp, vp))  # Add the edge to the path_edges set
            edge_midpoint = ((node_pos[last_vp][0] + node_pos[vp][0]) / 2, 
                             (node_pos[last_vp][1] + node_pos[vp][1]) / 2)
            # Place a text label on the edge
            plt.text(edge_midpoint[0], edge_midpoint[1], str(idx), color='green', fontsize=12)
        last_vp = vp
    
    if show_gt:
        last_vp = None
        for idx, vp in enumerate(gt_path):
            if last_vp:
                _G.add_edge(last_vp, vp)
                _gt_edges.append((last_vp, vp))  # Add the edge to the path_edges set
            last_vp = vp
    
        # gt
    if show_gt:
        nx.draw_networkx_edges(_G, pos=node_pos, ax=ax, edgelist=_gt_edges, edge_color=(1,0.5,1), alpha=0.2, width=3, arrowstyle='->')
        nx.draw_networkx_nodes(_G, pos=node_pos, ax=ax, node_size=80, nodelist=gt_path, node_color='k')
        
    # Drawing actual nodes
    nx.draw_networkx_nodes(_G, pos=node_pos, ax=ax, node_size=10, nodelist=set(_G.nodes()) - set(v['path']), node_color=(0.9,0.9,0.9))
    nx.draw_networkx_nodes(_G, pos=node_pos, ax=ax, node_size=50, nodelist=v['path'], node_color='b')
    

    # Drawing actual edges
    nx.draw_networkx_edges(_G, pos=node_pos, ax=ax, edgelist=set(_G.edges()) - set(_edges), edge_color=(0.9,0.9,0.9), alpha=1, width=1, arrows=False)
    nx.draw_networkx_edges(_G, pos=node_pos, ax=ax, edgelist=_edges, edge_color='b', alpha=1, width=2, arrowstyle='->')
    
    nx.draw_networkx_nodes(_G, pos=node_pos, ax=ax, node_size=50, nodelist=v['path'][:1], node_color='g')
    nx.draw_networkx_nodes(_G, pos=node_pos, ax=ax, node_size=50, nodelist=v['path'][-1:], node_color='r')
    
#     nx.draw_networkx_nodes(_G, pos=node_pos, ax=ax, node_size=50, nodelist=[kidnap_locs['curr_vp']], node_color='k')
    if kidnap_locs:
        nx.draw_networkx_nodes(_G, pos=node_pos, ax=ax, node_size=50, nodelist=[kidnap_locs['intended_vp']], node_color=(0.9,0.8,0))
        nx.draw_networkx_nodes(_G, pos=node_pos, ax=ax, node_size=50, nodelist=[kidnap_locs['kidnapped_vp']], node_color='k')
    
    
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.axis('off')
    plt.show()


def plot_gif(v, e, show_gt=False, kidnap_locs=None):
    instr_id = v['instr_id']
    gt = e.gt[int(instr_id.split('_')[0])]
    graph = e.graphs[gt['scan']]
    gt_path = gt['path']
    
    node_pos = nx.get_node_attributes(graph,'position')
    for k,vv in node_pos.items():
        node_pos[k] = vv[:-1]
    
    rel_pos = [node_pos[vp] for vp in v['path']]
    rel_x = [r[0] for r in rel_pos]
    rel_y = [r[1] for r in rel_pos]

    gt_rel_pos = [node_pos[vp] for vp in gt_path]
    gt_rel_x = [r[0] for r in gt_rel_pos]
    gt_rel_y = [r[1] for r in gt_rel_pos]
    
    rel_x.extend(gt_rel_x)
    rel_y.extend(gt_rel_y)
    
    xlim = [min(rel_x)-3, max(rel_x)+3]
    ylim = [min(rel_y)-3, max(rel_y)+3]
    
    _G = nx.DiGraph(graph)
    _nodes = set()

    fig, ax = plt.subplots()
    gt_path = gt['path']

    def init():
        pass
    
    def animate(idx):
#         ax.clear()
        _edges = []
        _gt_edges = []
        for i in range(idx):
            if i == 0:
                continue
            if i < len(v['path']):
                vp = v['path'][i]
                last_vp = v['path'][i - 1]
                _G.add_edge(last_vp, vp)
                _edges.append((last_vp, vp))
                edge_midpoint = ((node_pos[last_vp][0] + node_pos[vp][0]) / 2 + 0.1, 
                                 (node_pos[last_vp][1] + node_pos[vp][1]) / 2 + 0.1)
                ax.text(edge_midpoint[0], edge_midpoint[1], str(i), color='green', fontsize=15)
            if i < len(gt_path) and show_gt:
                gt_vp = gt_path[i]
                gt_last_vp = gt_path[i - 1]
                _G.add_edge(gt_last_vp, gt_vp)
                _gt_edges.append((gt_last_vp, gt_vp))

        if show_gt:
            nx.draw_networkx_edges(_G, pos=node_pos, ax=ax, edgelist=_gt_edges[:idx], edge_color=(1,0.5,1), alpha=1, width=2, arrowstyle='->')
            nx.draw_networkx_nodes(_G, pos=node_pos, ax=ax, node_size=150, nodelist=gt_path[1:idx], node_color='k')

        nx.draw_networkx_nodes(_G, pos=node_pos, ax=ax, node_size=50, nodelist=v['path'][:idx], node_color='b')
        
        if idx:
            nx.draw_networkx_nodes(_G, pos=node_pos, ax=ax, node_size=100, nodelist=v['path'][:1], node_color='r')
        if idx == len(v['path']):
            nx.draw_networkx_nodes(_G, pos=node_pos, ax=ax, node_size=100, nodelist=v['path'][-1:], node_color='g')
        
        nx.draw_networkx_edges(_G, pos=node_pos, ax=ax, edgelist=_edges[:-1], edge_color='b', alpha=1, width=3, arrows=True)
        nx.draw_networkx_edges(_G, pos=node_pos, ax=ax, edgelist=_edges[-1:], edge_color='r', alpha=1, width=3, arrows=True)
        
        if kidnap_locs:
            nx.draw_networkx_nodes(_G, pos=node_pos, ax=ax, node_size=50, nodelist=[kidnap_locs['intended_vp']], node_color=(0.9,0.8,0))
#             nx.draw_networkx_nodes(_G, pos=node_pos, ax=ax, node_size=50, nodelist=[kidnap_locs['kidnapped_vp']], node_color='k')


    if show_gt:
        t = max(len(v['path']), len(gt_path))
    else:
        t = len(v['path'])
    nx.draw_networkx_nodes(_G, pos=node_pos, ax=ax, node_size=50, nodelist=_G.nodes(), node_color=(0.9,0.9,0.9))
    nx.draw_networkx_edges(_G, pos=node_pos, ax=ax, edgelist=_G.edges(), edge_color=(0.5,0.5,0.5), alpha=1, width=1, arrows=False)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.axis('off')
    return matplotlib.animation.FuncAnimation(fig, animate, init_func=init, frames=t+1, repeat=False)