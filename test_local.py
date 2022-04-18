import networkx as nx
import time, logging

def erdos_renyi(n=10000, m=50000, seed=1234):
    return nx.gnm_random_graph(n, m, seed)

def scale_free(n=10000, m=5, seed=1234):
    return nx.barabasi_albert_graph(n, m, seed)

def small_world(n=10000, k=10, p=0.1, seed=1234):
    return nx.watts_strogatz_graph(n, k, p, seed)

def powerlaw_cluster(n=10000, m=5, p=0.8, seed=1234):
    return nx.powerlaw_cluster_graph(n, m, p, seed)

def soc_livejournal(f='/Users/sahiltyagi/Downloads/soc-LiveJournal1.txt'):
    G = nx.read_adjlist(f)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f'n {n} and m {m}')

if __name__ == '__main__':
    # G = erdos_renyi()
    # G = scale_free()
    # G = small_world()
    # G = powerlaw_cluster()
    # s1 = time.time()
    # sorted(nx.connected_components(G), key=len, reverse=True)
    # print(f'Connected components erdos-renyi {time.time() - s1}')
    # s1 = time.time()
    # nx.clustering(G)
    # print(f'Clustering erdos-renyi {time.time() - s1}')
    # s1 = time.time()
    # nx.pagerank(G)
    # print(f'Pagerank erdos-renyi {time.time() - s1}')
    # s1 = time.time()
    # nx.betweenness_centrality(G)
    # print(f'Between_centrality erdos-renyi {time.time() - s1}')

    soc_livejournal()