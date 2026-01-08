#!/usr/bin/env python3
import argparse
import subprocess
import random
import sys


def read_graph_n(path):
    """Read only the number of vertices n from the graph file."""
    with open(path) as f:
        toks = f.read().split()
    if len(toks) < 1:
        raise ValueError("Graph file is empty")
    return int(toks[0])

def read_coloring(path, n):
    """Read a coloring file: one color per line."""
    colors = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            colors.append(int(line))

    if len(colors) != n:
        raise ValueError(
            f"Coloring file has {len(colors)} entries, but graph has {n} vertices"
        )

    for c in colors:
        if c not in (0, 1, 2):
            raise ValueError("Coloring must use only colors {0,1,2}")

    return colors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verifier",
        required=True,
        help='e.g. "./zk_verifier --graph G.txt --rounds 200"'
    )
    ap.add_argument("--graph", required=True)
    ap.add_argument("--coloring", required=True)
    args = ap.parse_args()

    # Read graph size and coloring
    n = read_graph_n(args.graph)
    base_colors = read_coloring(args.coloring, n)
    rng = random.Random()

    # you can use this to generate a random coloring if you want to test the prover without a given coloring
    # base_colors = [rng.randrange(3) for _ in range(n)]



    # Apply one random permutation for the entire execution
    perm = [0, 1, 2]
    rng.shuffle(perm)
    colors = [perm[c] for c in base_colors]

    # Start verifier
    p = subprocess.Popen(
        args.verifier.split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    assert p.stdin and p.stdout

    # Interaction loop
    while True:
        line = p.stdout.readline()
        if not line:
            break

        line = line.strip()
        if line.startswith("CHALLENGE"):
            _, us, vs = line.split()
            u = int(us)
            v = int(vs)
            p.stdin.write(f"ANSWER {colors[u]} {colors[v]}\n")
            p.stdin.flush()

        elif line.startswith("ACCEPT") or line.startswith("REJECT"):
            print(line)
            break

        else:
            print(f"Unexpected verifier output: {line}")
            break

    p.terminate()

if __name__ == "__main__":
    main()
