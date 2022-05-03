import argparse
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
from word2qdm import *
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--corpus_path", default="../data/cats_dogs.txt")
parser.add_argument("--output_path", default="../dms/word2qdm.model")
parser.add_argument("--vocab_path", default="../dms/vocab_itos.txt")
parser.add_argument("--num_qubits", type=int, default=5)
parser.add_argument("--num_dm_qubits", type=int, default=2)
parser.add_argument("--layers", type=int, default=1)
parser.add_argument("--window_size", type=int, default=4)
parser.add_argument("--num_runs", type=int, default=3)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=1)
# TODO: works only for neg_samples = 1
parser.add_argument("--neg_samples", type=int, default=1)
parser.add_argument("--min_count", type=int, default=1)
parser.add_argument("--subsampling", type=float, default=1e-1)
parser.add_argument("--neg_table_size", type=float, default=10e2)

args = parser.parse_args()
corpus_path = args.corpus_path
output_path = args.output_path
vocab_path = args.vocab_path
num_qubits = args.num_qubits
num_dm_qubits = args.num_dm_qubits
layers = args.layers
window_size = args.window_size
num_runs = args.num_runs
num_epochs = args.num_epochs
lr = args.lr
batch_size = args.batch_size
neg_samples = args.neg_samples
min_count = args.min_count
subsampling = args.subsampling
neg_table_size = args.neg_table_size

vocab = Word2DMVocab(
    corpus_path=corpus_path,
    min_count=min_count,
    subsampling=subsampling,
    neg_table_size=neg_table_size,
)
vocab.build_vocab()
dataset = SkipGramDataset(
    corpus_path=corpus_path,
    vocab=vocab,
    window_size=window_size,
    neg_samples=neg_samples,
)
loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
vocab_size = len(vocab.itos)
model = Word2QDM(vocab_size, layers, num_qubits, num_dm_qubits)

best_costs = []
best_epochs = []
best_targets = []
best_contexts = []
loss_lists = []

for r in range(num_runs):
    best_cost = 100
    best_target_params = np.zeros((vocab_size, layers, num_qubits))
    best_context_params = np.zeros((vocab_size, layers, num_qubits))
    best_epoch = 0
    loss_list = []

    opt = optim.Adam(model.parameters(), lr=lr)
    for n in range(num_epochs):
        epoch_loss = 0
        for target_ids, context_ids, neg_ids in tqdm(loader):
            if target_ids is None:
                continue
            opt.zero_grad()
            loss = model.forward(target_ids, context_ids, neg_ids)
            loss.backward()
            opt.step()
            epoch_loss += loss
        loss_list.append(epoch_loss)

        # Keeps track of best parameters of a run
        if epoch_loss < best_cost:
            best_epoch = n
            best_cost = epoch_loss
            best_target_params = model.target_params.clone()
            best_context_params = model.context_params.clone()

        # Keep track of progress every 10 epochs
        if n == 0 or n % 10 == 9 or n == num_epochs - 1:
            print("Cost after {} epochs is {:.4f}".format(n + 1, epoch_loss))

    best_costs.append(best_cost)
    best_epochs.append(best_epoch)
    best_targets.append(best_target_params.clone())
    best_contexts.append(best_context_params.clone())
    loss_lists.append(loss_list)

dms = model.get_density_matrices(best_targets)
torch.save(dms, output_path)
with open(vocab_path, 'w') as f:
    for word in vocab.itos:
        f.write("%s; " % word)

# vocab.neg_table = None # free up memory
# dm_array = model.get_density_matrices().cpu().numpy()
# dms = DensityMatrices(vocab, dm_array)
# print("Trained density matrices.")
#
# dms.normalise()
# print("Normalised density matrices.")
#
# dms.save(output_path)
# print("Saved normalised density matrices.")
