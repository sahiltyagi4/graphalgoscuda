import networkx as nx
import time, logging
import cugraph as cx
import random

def erdos_renyi(n=10000, m=50000, seed=1234):
    return nx.gnm_random_graph(n, m, seed)

def scale_free(n=10000, m=5, seed=1234):
    return nx.barabasi_albert_graph(n, m, seed)

def small_world(n=10000, k=10, p=0.1, seed=1234):
    return nx.watts_strogatz_graph(n, k, p, seed)

def powerlaw_cluster(n=10000, m=5, p=0.8, seed=1234):
    return nx.powerlaw_cluster_graph(n, m, p, seed)

def real_graph(f='/home/styagi/com-youtube.ungraph.txt'):
    return nx.read_adjlist(f)

def amazon_graph(f='/home/styagi/com-amazon.ungraph.txt'):
    return nx.read_adjlist(f)

def connected_components(G, device='cpu'):
    strt_time = time.time()
    if device == 'cpu':
        nx.connected_components(G)
    else:
        cx.connected_components(G)
    return float(time.time() - strt_time)

def triangles(G, device='cpu'):
    strt_time = time.time()
    if device == 'cpu':
        nx.triangles(G)
    else:
        cx.triangles(G)
    return float(time.time() - strt_time)

def pagerank(G, device='cpu'):
    strt_time = time.time()
    if device == 'cpu':
        nx.pagerank(G)
    else:
        cx.pagerank(G)
    return float(time.time() - strt_time)

def betweenness_centrality(G, device='cpu'):
    strt_time = time.time()
    if device == 'cpu':
        nx.betweenness_centrality(G)
    else:
        cx.betweenness_centrality(G)
    return float(time.time() - strt_time)

if __name__ == '__main__':
    logging.basicConfig(filename='networkx_cugraph_perf-'+str(random.randint(10, 999))+'.log', level=logging.INFO)

    # G = real_graph()
    G = amazon_graph()
    t = connected_components(G, device='gpu')
    logging.info(f'SOC_LiveJournal graph connected_components CUDA {t} seconds')
    t = triangles(G, device='gpu')
    logging.info(f'SOC_LiveJournal graph clustering CUDA {t} seconds')
    t = pagerank(G, device='gpu')
    logging.info(f'SOC_LiveJournal graph Pagerank CUDA {t} seconds')
    t = betweenness_centrality(G, device='gpu')
    logging.info(f'SOC_LiveJournal graph Betweenness_centrality CUDA {t} seconds')
    logging.info('################################################')


    # num_nodes = 1000000
    # num_edges = 5000000
    # num_nodes = 100000
    # num_edges = 500000
    # numedges_pernode = int(num_edges/num_nodes)
    # G = erdos_renyi(n=num_nodes, m=num_edges)
    # t = connected_components(G, device='gpu')
    # logging.info(f'erdos-renyi graph connected_components CUDA {t} seconds')
    # t = triangles(G, device='gpu')
    # logging.info(f'erdos-renyi graph clustering CUDA {t} seconds')
    # t = pagerank(G, device='gpu')
    # logging.info(f'erdos-renyi graph Pagerank CUDA {t} seconds')
    # t = betweenness_centrality(G, device='gpu')
    # logging.info(f'erdos-renyi graph Betweenness_centrality CUDA {t} seconds')
    # logging.info('################################################')
    #
    # G = scale_free(n=num_nodes, m=numedges_pernode)
    # t = connected_components(G, device='gpu')
    # logging.info(f'barasbi-albert graph connected_components CUDA {t} seconds')
    # t = triangles(G, device='gpu')
    # logging.info(f'barasbi-albert graph clustering CUDA {t} seconds')
    # t = pagerank(G, device='gpu')
    # logging.info(f'barasbi-albert graph Pagerank CUDA {t} seconds')
    # t = betweenness_centrality(G, device='gpu')
    # logging.info(f'barasbi-albert graph Betweenness_centrality CUDA {t} seconds')
    # logging.info('################################################')
    #
    # G = small_world(n=num_nodes, k=numedges_pernode, p=0.2)
    # t = connected_components(G, device='gpu')
    # logging.info(f'watts-strogatz graph connected_components CUDA {t} seconds')
    # t = triangles(G, device='gpu')
    # logging.info(f'watts-strogatz graph clustering CUDA {t} seconds')
    # t = pagerank(G, device='gpu')
    # logging.info(f'watts-strogatz graph Pagerank CUDA {t} seconds')
    # t = betweenness_centrality(G, device='gpu')
    # logging.info(f'watts-strogatz graph Betweenness_centrality CUDA {t} seconds')
    # logging.info('################################################')
    #
    # G = powerlaw_cluster(n=num_nodes, m=numedges_pernode, p=0.5)
    # t = connected_components(G, device='gpu')
    # logging.info(f'powerlaw-cluster graph connected_components CUDA {t} seconds')
    # t = triangles(G, device='gpu')
    # logging.info(f'powerlaw-cluster graph clustering CUDA {t} seconds')
    # t = pagerank(G, device='gpu')
    # logging.info(f'powerlaw-cluster graph Pagerank CUDA {t} seconds')
    # t = betweenness_centrality(G, device='gpu')
    # logging.info(f'powerlaw-cluster graph Betweenness_centrality CUDA {t} seconds')