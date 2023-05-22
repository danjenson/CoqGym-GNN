from gallina import traverse_postorder
import torch

from .non_terminals import nonterminals


def create_edge_index(ast):
    "Creates an edge index for the given ast."
    index_map = {}
    counter = [0]

    def index_callbck(node):
        index_map[node.meta] = counter[-1]
        counter.append(counter[-1] + 1)

    traverse_postorder(ast, index_callbck)

    edge_index = []

    def callbck(node):
        for child in node.children:
            parent_child = [index_map[node.meta], index_map[child.meta]]
            child_parent = [index_map[child.meta], index_map[node.meta]]
            edge_index.append(parent_child)
            edge_index.append(child_parent)

    traverse_postorder(ast, callbck)

    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def create_x(ast):
    """
    Creates a feature set per nodes given the AST of the nodes as a lark tree.

    Shape: (num_nodes, )
    """
    x = []

    idents: list[tuple] = []

    def callbck(node):
        i = len(x)
        x.append([nonterminals.index(node.data)])
        if hasattr(node, "ident"):
            idents.append((i, node.ident))

    traverse_postorder(ast, callbck)

    return torch.tensor(x, dtype=torch.long), idents
