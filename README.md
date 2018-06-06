# tx-relay-simulation

Requires boost.

Builds (for me) with

```
g++ -O3 -g -std=c++11 -pthread -lboost_program_options graph.cpp -o graph
```

Example command line, for simulating bucketing of inbound peers:

```
./graph --simtype=2 --estimator=1 --buckets=2 --nodes 10000 --threads 64  --outbound 8 --trials 10000
```

To simulate using 4 buckets but further doubling the delay to inbound peers:

```
./graph --simtype=2 --estimator=1 --buckets=4 --nodes 10000 --threads 64  --outbound 8 --trials 10000 --inboundscale 0.25
```
