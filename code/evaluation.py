import argparse
import math
import torch


def similarity(word1_dm, word2_dm):
    # trace inner product
    # Efficient way to compute trace of matrix product
    trace = (word1_dm * word2_dm.T).sum().item()
    # Normalise
    trace = trace / (
        math.sqrt((word1_dm**2).sum()) * math.sqrt((word2_dm**2).sum())
    )
    return trace


parser = argparse.ArgumentParser()
parser.add_argument("--vocab_path", default="../dms/vocab_itos.txt")
parser.add_argument("--input_path", default="../dms/word2qdm.model")
parser.add_argument("--dataset", default="abc")

args = parser.parse_args()
vocab_path = args.vocab_path
input_path = args.input_path
dataset = args.dataset

dms_runs = torch.load(input_path)

with open(vocab_path, "r") as f:
    file_content = f.read()
vocab_itos = file_content.split("; ")[:-1]

ii_sim = []
ij_sim = []
for dms in dms_runs:
    if dataset == "animals":
        cats = dms[vocab_itos.index("cats")]
        dogs = dms[vocab_itos.index("dogs")]
        cuddling = dms[vocab_itos.index("cuddling")]
        alone = dms[vocab_itos.index("alone")]
        independent = dms[vocab_itos.index("independent")]
        depend = dms[vocab_itos.index("depend")]
        meow = dms[vocab_itos.index("meow")]
        bark = dms[vocab_itos.index("bark")]
        lions = dms[vocab_itos.index("lions")]
        wild = dms[vocab_itos.index("wild")]
        hunt = dms[vocab_itos.index("hunt")]

        ii_pairs = [
            (dogs, cuddling),
            (cats, alone),
            (dogs, depend),
            (cats, independent),
            (dogs, bark),
            (cats, meow),
            (lions, wild),
            (lions, hunt),
        ]
        ij_pairs = [
            (dogs, alone),
            (dogs, independent),
            (dogs, meow),
            (dogs, wild),
            (dogs, hunt),
            (cats, cuddling),
            (cats, depend),
            (cats, bark),
            (cats, wild),
            (cats, hunt),
            (lions, cuddling),
            (lions, alone),
            (lions, depend),
            (lions, independent),
            (lions, bark),
            (lions, meow),
        ]

    elif dataset == "abc":
        a1 = dms[vocab_itos.index("a1")]
        a2 = dms[vocab_itos.index("a2")]
        a3 = dms[vocab_itos.index("a3")]
        b1 = dms[vocab_itos.index("b1")]
        b2 = dms[vocab_itos.index("b2")]
        b3 = dms[vocab_itos.index("b3")]
        c1 = dms[vocab_itos.index("c1")]
        c2 = dms[vocab_itos.index("c2")]
        c3 = dms[vocab_itos.index("c3")]

        ii_pairs = [
            (a1, a2),
            (a1, a3),
            (a2, a3),
            (b1, b2),
            (b1, b3),
            (b2, b3),
            (c1, c2),
            (c1, c3),
            (c2, c3),
        ]
        ij_pairs = [
            (a1, b1),
            (a1, b2),
            (a1, b3),
            (a2, b1),
            (a2, b2),
            (a2, b3),
            (a3, b1),
            (a3, b2),
            (a3, b3),
            (a1, c1),
            (a1, c2),
            (a1, c3),
            (a2, c1),
            (a2, c2),
            (a2, c3),
            (a3, c1),
            (a3, c2),
            (a3, c3),
            (b1, c1),
            (b1, c2),
            (b1, c3),
            (b2, c1),
            (b2, c2),
            (b2, c3),
            (b3, c1),
            (b3, c2),
            (b3, c3),
        ]

    av_sim_ii = 0
    for pair in ii_pairs:
        av_sim_ii += similarity(pair[0], pair[1])
    av_sim_ii /= len(ii_pairs)
    ii_sim.append(av_sim_ii)

    av_sim_ij = 0
    for pair in ij_pairs:
        av_sim_ij += similarity(pair[0], pair[1])
    av_sim_ij /= len(ij_pairs)
    ij_sim.append(av_sim_ij)

print("ii_sim per run: \n", ii_sim)
print("ij_sim per run: \n", ij_sim)
print("average sim_ii: ", sum(ii_sim) / dms_runs.shape[0])
print("average sim_ij: ", sum(ij_sim) / dms_runs.shape[0])
