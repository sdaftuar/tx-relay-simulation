#include <iostream>
#include <set>
#include <utility>
#include <random>
#include <vector>
#include <algorithm>
#include <map>
#include <list>
#include <cmath>
#include <assert.h>
#include <limits>
#include <boost/math/distributions/gamma.hpp>
#include <thread>
#include <numeric>
#include <boost/program_options.hpp>

enum { BasicDiffusion = 1, BucketedDiffusion = 2 };
enum { FirstSpy = 1, MLSim = 2, MLShortestPath = 3 };

constexpr int DEFAULT_NODES = 1000;
constexpr int DEFAULT_INBOUND_NODES = 0; // nodes that are not listening
constexpr int DEFAULT_OUTBOUND = 6;
constexpr int DEFAULT_TRIALS = 1000;
constexpr int DEFAULT_SIMTYPE = BasicDiffusion;
constexpr int DEFAULT_ESTIMATOR = FirstSpy;
constexpr double DEFAULT_INBOUND_SCALE = 0.5;
constexpr int DEFAULT_THETA = 1;
constexpr int DEFAULT_THREADS = 8;
constexpr int DEFAULT_BUCKETS = 8;
constexpr bool DEFAULT_RELAYSTATS = false;

int g_num_threads = DEFAULT_THREADS;

struct Options {
    int num_nodes = DEFAULT_NODES; // listening nodes
    int num_inbound_nodes = DEFAULT_INBOUND_NODES; // non-listening nodes
    int num_outbound = DEFAULT_OUTBOUND;
    int num_trials = DEFAULT_TRIALS;
    // num_threads is a global, for now
    int simtype = DEFAULT_SIMTYPE;
    int estimator = DEFAULT_ESTIMATOR;
    double inbound_scale = DEFAULT_INBOUND_SCALE;
    int theta = DEFAULT_THETA; // only relevant for basic DiffusionSpreader
    int buckets = DEFAULT_BUCKETS; // only relevant for bucketed diffusion spreader
    bool relay_stats = DEFAULT_RELAYSTATS;

    void Print() {
        printf("Using %d listening nodes in graph\n", num_nodes);
        printf("Using %d non-listening nodes in graph\n", num_inbound_nodes);
        printf("Using %d outbound edges per node\n", num_outbound);
        printf("Running %d trials\n", num_trials);
        printf("Using %d threads\n", g_num_threads);
        printf("Setting simtype = %s\n", simtype == BasicDiffusion ? "basic diffusion" : "bucketed diffusion");
        printf("Setting estimator = %s\n", estimator == FirstSpy ? "first spy" : estimator == MLSim ? "ML simulation" : "MLShortestPath");
        printf("Using %f as scale parameter for inbound nodes in diffusion model\n", inbound_scale);
        if (simtype != BucketedDiffusion) {
            printf("Using adversary theta = %d\n", theta);
        } else {
            printf("Using %d inbound buckets for diffusion model\n", buckets);
        }
        printf("Will %scalculate propagation statistics\n", relay_stats ? "" : "not ");
    }
};

/*
 * TODO
 *
 * 1) add command line selection of:
 *    a) number of nodes / number of outbound connections [done]
 *    b) which diffusion model to use [done]
 *    c) how many trials to run [done]
 *    d) how many trials to run before creating a new graph [eh - now each thread uses a different graph]
 *    e) what the inbound relay delay should be [done]
 *    f) how many buckets if using the bucketed model [done]
 *    g) how many connections for the adversary if using a non-bucketed model [done]
 * 2) Add smarter estimators:
 *    a) maximum likelihood
 *    b) reporting centrality / rumor centrality?
 * 3) Replace graph implementation with boost graph or some other package so that we can
 *    use graph algorithms that are already written.
 * 4) Add statistics for total network propagation delays.
 * 5) Parallelize the slowest part into multiple threads. [half done]
 */

//-----------------------------------------------------------------------------
// Graph implementation
//
//

// Base implementation of a directed graph
// vertices are referenced by integer index
// (out-)edges are stored in a list for each vertex
class DirectedGraph {
public:
    DirectedGraph(int num_vertices) : m_edges(num_vertices) {}
    virtual ~DirectedGraph() {}

    void AddEdge(size_t source, size_t target, int weight);
    int GetEdgeWeight(size_t source, size_t target);

    typedef std::set<size_t> vertex_set;

    // Add lookup functions (like getting neighbors?)
    vertex_set GetNeighbors(size_t source);
    size_t NumNodes() const { return m_edges.size(); }

public:
    // m_edges[source] is a set of outbound edges from a given source node.
    // each outbound edge has a target node and weight.
    std::vector<std::set<std::pair<size_t, int>>> m_edges;
};

void DirectedGraph::AddEdge(size_t source, size_t target, int weight)
{
    m_edges[source].insert(std::make_pair(target, weight));
}

int DirectedGraph::GetEdgeWeight(size_t source, size_t target)
{
    for (auto it=m_edges[source].begin(); it != m_edges[source].end(); ++it) {
        if (it->first == target) return it->second;
    }
    return 0;
}

DirectedGraph::vertex_set DirectedGraph::GetNeighbors(size_t node)
{
    vertex_set ret;
    for (auto it=m_edges[node].begin(); it !=m_edges[node].end(); ++it) {
        ret.insert(it->first);
    }
    return ret;
}

// RandomGraph: a directed graph that is designed to look like something that
// might occur on the bitcoin network.
// Create num_vertices random nodes.
// For each node, pick num_outbound other nodes at random to be the outbound
// peers.
// As this is a directed graph, we create edges in each direction when a new
// connection is made, but with different weights so that we can distinguish
// the inbound from the outbound edges.
class RandomGraph : public DirectedGraph {
public:
    RandomGraph(int num_vertices, int num_outbound, int num_inbound_only);
    virtual ~RandomGraph() {}
};

RandomGraph::RandomGraph(int num_vertices, int num_outbound, int num_inbound_only)
    : DirectedGraph(num_vertices+num_inbound_only)
{
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<unsigned int> node_indices;

    // Put all the listening nodes in a vector, which we'll randomly permute
    // and draw from
    for (unsigned int i=0; i<num_vertices; ++i) {
        node_indices.push_back(i);
    }

    // For each node, listening or not, we'll select num_outbound peers from
    // the listening set.
    for (unsigned int i=0; i<num_vertices+num_inbound_only; ++i) {
        std::shuffle(node_indices.begin(), node_indices.end(), g);
        int edges = 0;
        auto vertex_it = node_indices.begin();
        while (edges < num_outbound) {
            if (*vertex_it != i) {
                // The bitcoin p2p layer will communicate bidirectionally,
                // implying that we would use an undirected graph to model
                // communication. But because the bitcoin p2p logic
                // distinguishes inbound peers from outbound peers (eg for
                // transaction relay delays), we instead use a directed graph,
                // but create two edges with different weights between each
                // node-pair to distinguish the inbound connection from the
                // outbound.
                AddEdge(i, *vertex_it, 1);
                AddEdge(*vertex_it, i, 2);
                ++edges;
            }
            ++vertex_it;
        }
    }
}

// Helper function to print out the edges in the graph.
void PrintGraph(const RandomGraph &g)
{
    for (size_t i=0; i<g.NumNodes(); ++i) {
        printf("%lu: ", i);
        for (auto it=g.m_edges[i].begin(); it!=g.m_edges[i].end(); ++it) {
            printf("[%lu (%d)] ", it->first, it->second);
        }
        printf("\n");
    }
}

//-----------------------------------------------------------------------------
// Spreading models.

// DiffusionSpreader:

// Given a DirectedGraph and a source node, we model the propagation of a
// message from that source node to each other node on the network, and we
// record the timestamps that an adversary node might receive those
// announcements from each honest node. 
// Propagation between honest nodes simulates the transaction relay system on
// the Bitcoin network, where we use exponentially distributed random delays
// for each peer. Inbound peers sample from a distribution with twice the mean
// delay of outbound peers.
// Adversary is modeled as having theta inbound connections to every node.

class DiffusionSpreader {
public:
    // source == source node to use for spreading the message (if out of range,
    // a random source will be chosen)
    // theta == number of (inbound) connections the adversary has to each node.
    DiffusionSpreader(const DirectedGraph &g, size_t source=0, double theta=1.0, double inbound_multiplier=0.5);

    // Simulate spreading a message across the graph, and record the timestamp
    // at which each node and the adversary receive each message.  (Only the
    // adversary timestamps should be available to the estimators, but we use
    // the received timestamps internally for simulation.)
    void SpreadMessage();

    // Reset the state so that we can simulate spreading again from a new
    // source on the same graph.
    virtual void Reset(size_t new_source);

    // Pick a broadcast time for sending from a given source to target.
    // (source, target) must be an edge in the graph.
    // Distribution used is based on the weight of the edge, reflecting
    // the exponential delays used by Bitcoin Core.
    virtual double GetBroadcastTime(size_t source, size_t target, bool inbound);
    virtual double GetAdversaryTime(size_t source);

    const DirectedGraph &m_graph;
    size_t m_source;
    double m_theta;
    std::vector<double> adversary_timestamps;
    std::vector<double> received_timestamps;

    // Track the number of times each node is the first to broadcast a
    // transaction.
    std::vector<int> first_broadcasts;

    // Random distributions (is this right?)
    std::random_device rd;
    std::mt19937 generator;
    std::exponential_distribution<double> outbound_distribution;
    std::exponential_distribution<double> inbound_distribution;
    std::exponential_distribution<double> adversary_distribution;
};

// If the given source is out-of-range, randomly choose a source.
DiffusionSpreader::DiffusionSpreader(const DirectedGraph &g, size_t source, double theta, double inbound_multiplier)
    : m_graph(g), m_source(source), m_theta(theta), generator(rd()), outbound_distribution(1.0),
      inbound_distribution(inbound_multiplier), adversary_distribution(m_theta*inbound_multiplier)
{
    if (m_source > m_graph.NumNodes()) {
        m_source = rd() % m_graph.NumNodes();
    }
    adversary_timestamps.resize(m_graph.NumNodes(), -1);
    received_timestamps.resize(m_graph.NumNodes(), -1);
    first_broadcasts.resize(m_graph.NumNodes(), 0);
}

double DiffusionSpreader::GetBroadcastTime(size_t source, size_t target, bool inbound)
{
    double ret = received_timestamps[source];
    if (inbound) {
        ret += inbound_distribution(generator);
    } else {
        ret += outbound_distribution(generator);
    }
    return ret;
}

double DiffusionSpreader::GetAdversaryTime(size_t source)
{
    return received_timestamps[source] + adversary_distribution(generator);
}

void DiffusionSpreader::Reset(size_t source)
{
    m_source = source;
    if (m_source > m_graph.NumNodes()) {
        std::random_device rd;
        m_source = rd() % m_graph.NumNodes();
    }
    for (size_t i=0; i<received_timestamps.size(); ++i) {
        adversary_timestamps[i] = -1;
        received_timestamps[i] = -1;
        first_broadcasts[i] = 0;
    }
}

void DiffusionSpreader::SpreadMessage()
{
    received_timestamps[m_source] = 0.;
    adversary_timestamps[m_source] = GetAdversaryTime(m_source);

    std::set<size_t> infected_nodes = { m_source };

    // Some data structures for managing which edges fire.
    typedef std::multimap<double, std::pair<size_t, size_t>> edge_broadcast_map;
    edge_broadcast_map active_edges;

    // After a node is infected once, it never needs to be infected again --
    // keep track of the outstanding broadcasts TO a node so that they can be
    // removed after infection.
    typedef std::map<size_t, std::list<edge_broadcast_map::iterator> > target_node_map;
    target_node_map tn_map;

    // Add initial edges to active_edges
    for (auto it = m_graph.m_edges[m_source].begin(); it != m_graph.m_edges[m_source].end(); ++it) {
        // if the edge weight is 2, then it's an inbound connection.
        double broadcast_time = GetBroadcastTime(m_source, it->first, it->second == 2);
        edge_broadcast_map::iterator inserted_it = active_edges.insert(std::make_pair(broadcast_time, std::make_pair(m_source, it->first)));
        // printf("inserted edge: %f, %lu -> %lu\n", inserted_it->first, inserted_it->second.first, inserted_it->second.second);

        tn_map[it->first].push_back(inserted_it);
    }

    while (!active_edges.empty()) {
        auto it = active_edges.begin(); // sorted by fire-time, so just look at
                                        // the first entry
        size_t source_node = it->second.first;
        size_t target_node = it->second.second;
        ++first_broadcasts[source_node];
        assert (!infected_nodes.count(target_node));
        // printf("Infecting %lu at %f\n", target_node, it->first);
        infected_nodes.insert(target_node);
        received_timestamps[target_node] = it->first;
        adversary_timestamps[target_node] = GetAdversaryTime(target_node);

        // Clean out no-longer-relevant edges
        for (auto dupe_edge_it = tn_map[target_node].begin(); 
                dupe_edge_it != tn_map[target_node].end(); ++dupe_edge_it) {
            active_edges.erase(*dupe_edge_it);
        }
        tn_map[target_node].clear();

        // Add new edges to fire
        for (auto edge_it = m_graph.m_edges[target_node].begin(); edge_it != m_graph.m_edges[target_node].end(); ++edge_it) {
            if (!infected_nodes.count(edge_it->first)) {
                double broadcast_time = GetBroadcastTime(target_node, edge_it->first, edge_it->second == 2);
                edge_broadcast_map::iterator inserted_it = active_edges.insert(std::make_pair(broadcast_time, std::make_pair(target_node, edge_it->first)));
                tn_map[edge_it->first].push_back(inserted_it);
            }
        }
    }
}

// Here we model bucketing the inbound peers.  We assume the adversary will use
// all the buckets, so theta==num_buckets.
// The theta that is passed in to the underlying DiffusionSpreader will be
// unused -- we model the adversary's time by looking at the minimum time
// across all inbound buckets.
class DiffusionSpreaderBucketedInbound : public DiffusionSpreader {
public:
    DiffusionSpreaderBucketedInbound(const DirectedGraph &g, 
            size_t num_inbound_buckets, size_t source=0, 
            double inbound_multiplier=0.5) :
        DiffusionSpreader(g, source, 1.0, inbound_multiplier),
        m_buckets(num_inbound_buckets) 
    { }

    virtual ~DiffusionSpreaderBucketedInbound() {}
    virtual void Reset(size_t new_source);

    virtual double GetBroadcastTime(size_t source, size_t target, bool inbound);
    virtual double GetAdversaryTime(size_t source);

    size_t m_buckets;
    std::map<size_t, std::vector<double>> inbound_broadcast_map;
};

void DiffusionSpreaderBucketedInbound::Reset(size_t source)
{
    DiffusionSpreader::Reset(source);
    inbound_broadcast_map.clear();
}

// The first time we try to get a time for an inbound peer, we populate the
// inbound-peer buckets.  We pick a bucket based on the target's index value.
double DiffusionSpreaderBucketedInbound::GetBroadcastTime(size_t source, size_t target, bool inbound)
{
    if (inbound) {
        if (inbound_broadcast_map.count(source) == 0) {
            inbound_broadcast_map[source].resize(m_buckets);
            // Initialize the buckets
            for (size_t bucket=0; bucket<m_buckets; ++bucket) {
                inbound_broadcast_map[source][bucket] = DiffusionSpreader::GetBroadcastTime(source, target, true);
            }
        }
        return inbound_broadcast_map[source][target%m_buckets];
    }
    return DiffusionSpreader::GetBroadcastTime(source, target, false);
}

// We assume that the adversary is using all the buckets.
double DiffusionSpreaderBucketedInbound::GetAdversaryTime(size_t source)
{
    // Initialize the inbound map if necessary
    GetBroadcastTime(source, 0, true);
    double earliest_time = 99999;
    for (size_t bucket=0; bucket<m_buckets; ++bucket) {
        if (inbound_broadcast_map[source][bucket] < earliest_time) {
            earliest_time = inbound_broadcast_map[source][bucket];
        }
    }
    return earliest_time;
}

//-----------------------------------------------------------------------------
// Random helpers that are probably no longer needed

void Subgraph(const DirectedGraph &g, size_t node)
{
    // Print out the nodes connected to given node
    int count=0;
    std::set<size_t> neighbors;
    std::set<size_t> traveled = { node };
    for (auto it=g.m_edges[node].begin(); it != g.m_edges[node].end(); ++it) {
        neighbors.insert(it->first);
    }
    while (!neighbors.empty()) {
        auto it = neighbors.begin();
        printf("[%lu] ", *it);
        ++count;
        traveled.insert(*it);
        for (auto next_it=g.m_edges[*it].begin(); next_it != g.m_edges[*it].end(); ++next_it) {
            if (!traveled.count(next_it->first)) {
                neighbors.insert(next_it->first);
            }
        }
        neighbors.erase(it);
    }
    printf("found %d nodes\n", count);
}

void PrintAdversaryTimestamps(const DiffusionSpreader &diffusion_spread)
{
    std::multimap<double, size_t> adversary_timestamps;

    for (size_t i=0; i<diffusion_spread.adversary_timestamps.size(); ++i) {
        adversary_timestamps.insert(std::make_pair(diffusion_spread.adversary_timestamps[i], i));
        //printf("%lu %.3f %.3f\n", i, diffusion_spread.received_timestamps[i], diffusion_spread.adversary_timestamps[i]);
    }
    for (auto it=adversary_timestamps.begin(); it != adversary_timestamps.end(); ++it) {
        printf("%lu %.3f\n", it->second, it->first);
    }

    for (size_t i=0; i<diffusion_spread.received_timestamps.size(); ++i) {
        if (diffusion_spread.received_timestamps[i] == -1.) {
            Subgraph(diffusion_spread.m_graph, i);
        }
    }
}

//-----------------------------------------------------------------------------
// Estimators
//

// FirstSpyEstimator picks the first node to announce a message as the
// originator.
class FirstSpyEstimator {
public:
    FirstSpyEstimator(const DiffusionSpreader &diffusion_spread) : m_spreader(diffusion_spread) {}

    size_t EstimateSource();
    const DiffusionSpreader &m_spreader;
};

size_t FirstSpyEstimator::EstimateSource()
{
    std::multimap<double, size_t> adversary_timestamps; // sorted by time
    for (size_t i=0; i<m_spreader.adversary_timestamps.size(); ++i) {
        adversary_timestamps.insert(std::make_pair(m_spreader.adversary_timestamps[i], i));
    }
    return adversary_timestamps.begin()->second;
}

// MLSimEstimator
// This estimator simulates the given graph 1000 times for each source node, and records the observed
// timestamps in some buckets.
// When evaluating a given set of adversary timestamps, we look up the
// probability from each bucket and multiply them all (incorrectly assuming
// independence) and pick the source node that maximizes probability.
class MLSimEstimator {
public:
    MLSimEstimator(DiffusionSpreader &diffusion_spread);

    size_t EstimateSource();
    DiffusionSpreader &m_spreader;

    double Score(size_t candidate);

    // Just do 1/2 second buckets for now
    typedef std::map<size_t, std::map<size_t, std::map<int, double>>> source_dist_map;
    source_dist_map sim_distribution;
};

MLSimEstimator::MLSimEstimator(DiffusionSpreader &diffusion_spread)
    : m_spreader(diffusion_spread)
{
    int num_trials = 10000;
    for (size_t source=0; source<m_spreader.m_graph.NumNodes(); ++source) {
        std::map<size_t, std::list<double>> results;
        for (int i=0; i<num_trials; ++i) {
            m_spreader.Reset(source);
            m_spreader.SpreadMessage();
            // Record the results
            for (size_t target=0; target<m_spreader.m_graph.NumNodes(); ++target) {
                results[target].push_back(m_spreader.adversary_timestamps[target]);
            }
        }
        // Consolidate the results into a distribution for each target.
        for (size_t target=0; target < m_spreader.adversary_timestamps.size(); ++target) {
            printf("src %lu tgt %lu ", source, target);
            for (auto p : results[target]) {
                ++sim_distribution[source][target][p*2];
            }
            for (auto it = sim_distribution[source][target].begin(); it != sim_distribution[source][target].end(); ++it) {
                it->second /= num_trials;
                printf("[%.2f] ", it->second);
            }
            printf("\n");
        }
    }
}

double MLSimEstimator::Score(size_t candidate)
{
    double ret = 0;

    for (size_t target=0; target<m_spreader.adversary_timestamps.size(); ++target) {
        double value = m_spreader.adversary_timestamps[target];
        double prob = sim_distribution[candidate][target][2*value];
        if (prob == 0) prob = 1e-8;
        ret += std::log(prob);
    }
    return ret;
}

size_t MLSimEstimator::EstimateSource()
{
    std::map<double, size_t> scores;

    for (size_t candidate=0; candidate<m_spreader.m_graph.NumNodes(); ++candidate) {
        scores[Score(candidate)] = candidate;
    }
    return scores.rbegin()->second;
}

class MLShortestPathGammaEstimator {
public:
    MLShortestPathGammaEstimator(const DiffusionSpreader &diffusion_spread);
    MLShortestPathGammaEstimator(const DiffusionSpreader &diffusion_spread, const MLShortestPathGammaEstimator &use_paths);

    size_t EstimateSource();
    const DiffusionSpreader &m_spreader;

    void AssignSPLengthCalculations(std::list<size_t> source_list);
    void CalculateShortestPathLengths(size_t source, std::vector<int> &lengths);

    // Precalculate the shortest path length for each source node
    // path_lengths[source][target] is length of shortest path from source to
    // target.
    std::vector<std::vector<int>> path_lengths;
};

void MLShortestPathGammaEstimator::AssignSPLengthCalculations(std::list<size_t> source_list)
{
    for (auto p : source_list) {
        CalculateShortestPathLengths(p, path_lengths[p]);
    }
}

void MLShortestPathGammaEstimator::CalculateShortestPathLengths(size_t source, std::vector<int> &lengths)
{
    std::map<size_t, int> unvisited_distances;
    for (size_t target=0; target < m_spreader.m_graph.NumNodes(); ++target) {
        unvisited_distances[target] = std::numeric_limits<int>::max();
    }
    unvisited_distances[source] = 0;
    size_t current = source;
    while (true) {
        for (auto it=m_spreader.m_graph.m_edges[current].begin(); it != m_spreader.m_graph.m_edges[current].end(); ++it) {
            if (unvisited_distances.count(it->first) && unvisited_distances[it->first] > unvisited_distances[current]+it->second) {
                unvisited_distances[it->first] = unvisited_distances[current] + it->second;
            }
        }
        lengths[current] = unvisited_distances[current];
        unvisited_distances.erase(current);
        int min_distance = std::numeric_limits<int>::max();
        for (auto it = unvisited_distances.begin(); it != unvisited_distances.end(); ++it) {
            if (it->second < min_distance) {
                min_distance = it->second;
                current = it->first;
            }
        }
        if (min_distance == std::numeric_limits<int>::max()) {
            break;
        }
    }
    printf(".");
}

MLShortestPathGammaEstimator::MLShortestPathGammaEstimator(const DiffusionSpreader &diffusion_spread)
    : m_spreader(diffusion_spread), path_lengths(diffusion_spread.m_graph.NumNodes())
{
    // Use dijkstra's algorithm to find shortest path lengths
    std::vector<std::list<size_t>> jobs;
    jobs.resize(g_num_threads);
    for (size_t source=0; source < m_spreader.m_graph.NumNodes(); ++source) {
        //if (source % 100 == 0) printf("source: %lu\n", source);
        path_lengths[source].resize(path_lengths.size());
        jobs[source%g_num_threads].push_back(source);
    }
    std::thread threads[g_num_threads];
    for (int i=0; i<g_num_threads; ++i) {
        threads[i] = std::thread(&MLShortestPathGammaEstimator::AssignSPLengthCalculations, this, jobs[i]);
    }
    for (int i=0; i<g_num_threads; ++i) {
        threads[i].join();
    }
    printf("\n");
}

MLShortestPathGammaEstimator::MLShortestPathGammaEstimator(const DiffusionSpreader &diffusion_spread, const MLShortestPathGammaEstimator &use_paths)
    : m_spreader(diffusion_spread), path_lengths(use_paths.path_lengths)
{ 
    /*
    for (size_t i=0; i<path_lengths.size(); ++i) {
        printf("src %lu: ", i);
        for (size_t j=0; j<path_lengths[i].size(); ++j) {
            printf("[%d] ", path_lengths[i][j]);
        }
        printf("\n");
    }*/
}

size_t MLShortestPathGammaEstimator::EstimateSource()
{
    std::multimap<double, size_t> scores;

    for (size_t candidate=0; candidate<m_spreader.m_graph.NumNodes(); ++candidate) {
        double score = 0;
        for (size_t target=0; target < m_spreader.adversary_timestamps.size(); ++target) {
            double timestamp = m_spreader.adversary_timestamps[target];
            boost::math::gamma_distribution<> gamma_dist(path_lengths[candidate][target]+2);
            double pdf_value = boost::math::pdf(gamma_dist, timestamp);
            score += std::log(pdf_value);
        }
        //printf("%lu: %f\n", candidate, score);
        scores.insert(std::make_pair(score, candidate));
    }
    /*
    int count=0;
    for (auto it=scores.rbegin(); it != scores.rend(); ++it) {
        count++;
        //printf("%lu %f [%lu]\n", it->second, it->first, m_spreader.m_source);
        if (count > 9) break;
    }*/
    return scores.rbegin()->second;
}

void RunSimulation(const MLShortestPathGammaEstimator &has_paths, const DirectedGraph &g, int num_trials, int *successes)
{
    DiffusionSpreader diffusion_spread(g);
    MLShortestPathGammaEstimator ml_est(diffusion_spread, has_paths);
    *successes = 0;
    for (int i=0; i<num_trials; ++i) {
        diffusion_spread.Reset(-1);
        diffusion_spread.SpreadMessage();
        //printf("actual source: %lu\n", diffusion_spread.m_source);
        if (ml_est.EstimateSource() == diffusion_spread.m_source) {
            ++(*successes);
        }
        printf(".");
    }
}

void RunDiffusionSimulation(int trials, const Options &opt, int *success_count, std::map<double, std::list<double>> * propagation_delay = nullptr, std::map<int, int> * relay_histogram = nullptr)
{
    RandomGraph graph(opt.num_nodes, opt.num_outbound, opt.num_inbound_nodes);
    DiffusionSpreaderBucketedInbound *ds_bucket = nullptr;
    DiffusionSpreader *ds = nullptr;

    if (opt.simtype == BucketedDiffusion) {
        ds_bucket = new DiffusionSpreaderBucketedInbound(graph, opt.buckets, -1, opt.inbound_scale);
    } else {
        ds = new DiffusionSpreader(graph, -1, (double) opt.theta, opt.inbound_scale);
    }
    DiffusionSpreader &diffusion_spread = ds_bucket != nullptr ? *ds_bucket : *ds;
    *success_count = 0;

    // The ML simulators do graph analysis once, which is then used on each
    // simulation run. Precompute. TODO: move this to the caller.

    MLSimEstimator *ml_sim = nullptr;
    MLShortestPathGammaEstimator *ml_sp = nullptr;

    if (opt.estimator == MLSim) {
        ml_sim = new MLSimEstimator(diffusion_spread);
    } else if (opt.estimator == MLShortestPath) {
        ml_sp = new MLShortestPathGammaEstimator(diffusion_spread);
    }

    for (int i=0; i<trials; ++i) {
        diffusion_spread.Reset(-1); // clear state and pick a random new source node
        diffusion_spread.SpreadMessage();
        if (propagation_delay != nullptr) {
            std::vector<double> data(diffusion_spread.received_timestamps);
            std::sort(data.begin(), data.end());
            size_t num_elements = data.size();
            for (auto it = propagation_delay->begin(); it != propagation_delay->end(); ++it) {
                double pct = it->first;
                int lookup = std::max(0, int(num_elements*pct - 1));
                it->second.push_back(data[lookup]);
            }
        }
        if (relay_histogram != nullptr) {
            for (size_t i=0; i<diffusion_spread.first_broadcasts.size(); ++i) {
                ++(*relay_histogram)[diffusion_spread.first_broadcasts[i]];
            }
        }
        size_t estimate = -1;

        if (opt.estimator == FirstSpy) {
            FirstSpyEstimator first_spy(diffusion_spread);
            estimate = first_spy.EstimateSource();
        } else if (opt.estimator == MLSim) {
            estimate = ml_sim->EstimateSource();
        } else {
            estimate = ml_sp->EstimateSource();
        }
        if (estimate == diffusion_spread.m_source) {
            ++(*success_count);
        }
    }
}

// Run a diffusion simulation, in parallel
void LaunchDiffusionSim(const Options& opt)
{
    std::thread t[g_num_threads];
    std::vector<int> num_successes(g_num_threads);
    std::map<int, std::map<double, std::list<double>>> propagation_data;
    // Keep a histogram of the number of nodes that broadcasted a transaction a
    // given number of times.
    // relay_histogram: <relay count> --> <number of nodes>
    std::map<int, int> relay_histogram[g_num_threads];

    for (int i=0; i<g_num_threads; ++i) {
        if (opt.relay_stats) {
            propagation_data[i][0.05];
            propagation_data[i][0.10];
            propagation_data[i][0.50];
            propagation_data[i][0.75];
            propagation_data[i][0.95];
            propagation_data[i][0.99];
            propagation_data[i][1.00];
        }
        t[i] = std::thread(&RunDiffusionSimulation, opt.num_trials/g_num_threads, std::ref(opt), &num_successes[i], opt.relay_stats ? &propagation_data[i] : nullptr, opt.relay_stats ? &relay_histogram[i] : nullptr);
    }

    for (int i=0; i<g_num_threads; ++i) {
        t[i].join();
    }

    // Tally results
    int total_success = 0;
    for (auto s : num_successes) {
        total_success += s;
    }
    int num_runs = g_num_threads * (opt.num_trials / g_num_threads);
    printf("success %d / %d = %.2f\n", total_success, num_runs, double(total_success)/num_runs);

    // Tally relay statistics. Output mean time to reach each fraction of nodes.
    if (opt.relay_stats) {
        std::map<int, int> overall_histogram;
        for (size_t i=0; i<g_num_threads; ++i) {
            for (auto it = relay_histogram[i].begin(); it != relay_histogram[i].end(); ++it) {
                overall_histogram[it->first] += it->second;
            }
        }
        printf("Relay count histogram\n");
        for (auto it=overall_histogram.begin(); it != overall_histogram.end(); ++it) {
            printf("%d %d\n", it->first, it->second);
        }
        std::map<double, double> summary;
        for (auto map_it = propagation_data.begin(); map_it != propagation_data.end(); ++map_it) {
            for (auto it = map_it->second.begin(); it != map_it->second.end(); ++it) {
                assert(it->second.size() == opt.num_trials / g_num_threads);
                if (summary.count(it->first) == 0) summary[it->first] = 0;
                summary[it->first] += std::accumulate(it->second.begin(), it->second.end(), 0.);
            }
        }
        printf("Relay distribution - mean time to reach each percentile of nodes:\n");
        for (auto it = summary.begin(); it != summary.end(); ++it) {
            it->second /= num_runs;
            printf("%.2f %f\n", it->first, it->second);
        }
    }
}

int ParseArguments(int ac, char **av, Options &options)
{
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("nodes", po::value<int>(), "set number of listening nodes in graph")
        ("inbound_only", po::value<int>(), "set number of non-listening nodes in graph")
        ("outbound", po::value<int>(), "set number of outbound edges per node")
        ("trials", po::value<int>(), "set number of simulations to run")
        ("simtype", po::value<int>(), "pick a spreading model:\n"
                                      "\t1 == basic diffusion\n"
                                      "\t2 == bucketed diffusion\n")
        ("estimator", po::value<int>(), "pick an estimator:\n"
                                        "\t1 == first spy estimator\n"
                                        "\t2 == ML simulation estimator\n"
                                        "\t3 == ML shortest-path-gamma estimator\n")
        ("inboundscale", po::value<double>(), "pick an inbound scaling constant (default 0.5)")
        ("adversarytheta", po::value<int>(), "set number of adversary connections to each node (for basic diffusion only)") 
        ("buckets", po::value<int>(), "set number of timing buckets for inbound peer tx relay (for bucketed diffusion only)") 
        ("threads", po::value<int>(), "set number of threads to use")
        ("relaystats", po::value<bool>(), "set whether to output distribution statistics");
    ;

    try {
    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("nodes")) {
        options.num_nodes = vm["nodes"].as<int>();
    }

    if (vm.count("inbound_only")) {
        options.num_inbound_nodes = vm["inbound_only"].as<int>();
    }
    if (vm.count("outbound")) {
        options.num_outbound = vm["outbound"].as<int>();
    }

    if (vm.count("trials")) {
        options.num_trials = vm["trials"].as<int>();
    }

    if (vm.count("threads")) {
        g_num_threads = vm["threads"].as<int>();
    }

    if (vm.count("simtype")) {
        options.simtype = vm["simtype"].as<int>();
    }

    if (vm.count("estimator")) {
        options.estimator = vm["estimator"].as<int>();
    } 

    if (vm.count("inboundscale")) {
        options.inbound_scale = vm["inboundscale"].as<double>();
    }

    if (vm.count("adversarytheta")) {
        if (options.simtype == BucketedDiffusion) {
            printf("--adversarytheta not applicable to BucketedDiffusion! (see --help)\n");
            return 1;
        }
        options.theta = vm["adversarytheta"].as<int>();
    }
    if (vm.count("buckets")) {
        if (options.simtype == BasicDiffusion) {
            printf("--buckets not applicable to basic diffusion! (see --help)\n");
            return 1;
        }
        options.buckets = vm["buckets"].as<int>();
    }
    if (vm.count("relaystats")) {
        options.relay_stats = vm["relaystats"].as<bool>();
    }
    } catch(...) {
        printf("Unhandled exception (misparsed argument?), exiting\n");   
        return 1;
    }
    return 0;
}

int main(int ac, char* av[])
{
    Options opt;
    if (ParseArguments(ac, av, opt)) { return 1; }

    opt.Print();

    // Create a random graph to use for our simulations.
    //RandomGraph g(opt.num_nodes, opt.num_outbound);
    //PrintGraph(g);

    int num_correct = 0;

    LaunchDiffusionSim(opt);

    return 0;
}
