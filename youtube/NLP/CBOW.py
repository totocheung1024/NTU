import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils import process_w2v_data
from visual import show_w2v_word_embedding 
corpus = [
    # numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # alphabets, expecting that 9 is close to letters
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]


class CBOW(nn.Module):
    def __init__(self, v_dim, emb_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(v_dim, emb_dim)
        # Noise contrastive estimation loss initialization
        self.nce_w = nn.Parameter(torch.randn(v_dim, emb_dim))
        self.nce_b = nn.Parameter(torch.zeros(v_dim))

    def forward(self, x):
        x = self.embeddings(x)
        x = torch.mean(x, dim=1)
        return x

def nce_loss(inputs, targets, nce_w, nce_b, num_sampled, v_dim):
    batch_size, emb_dim = inputs.size()
    noise_dist = torch.ones(v_dim)
    total_loss = 0.0

    for i in range(batch_size):
        # Compute noise sample for each target word in the batch
        noise_samples = torch.multinomial(noise_dist, num_sampled, replacement=True)

        # Compute scores for positive sample
        true_score = torch.dot(inputs[i], nce_w[targets[i]]) + nce_b[targets[i]]

        # Compute scores for negative samples
        neg_scores = torch.mv(nce_w[noise_samples], inputs[i]) + nce_b[noise_samples]

        # Apply the logistic function to compute loss
        true_loss = F.logsigmoid(true_score)
        neg_loss = torch.sum(F.logsigmoid(-neg_scores))
        total_loss += -(true_loss + neg_loss)
    return total_loss / batch_size

def train(model, data, optimizer, num_sampled):
    model.train()
    total_loss = 0
    for t in range(6000):
        bx, by = data.sample(8)
        bx = torch.LongTensor(bx)
        by = torch.LongTensor(by)
        
        optimizer.zero_grad()
        output = model(bx)
        loss = nce_loss(output, by, model.nce_w, model.nce_b, num_sampled, data.num_word)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if t % 200 == 0:
            print("step: {} | loss: {}".format(t, total_loss / (t + 1)))

if __name__ == "__main__":
    d = process_w2v_data(corpus, skip_window=2, method="cbow")
    m = CBOW(d.num_word, 2)
    optimizer = optim.Adam(m.parameters(), lr=0.01)
    train(m, d, optimizer, num_sampled=5)

    # plotting
    show_w2v_word_embedding(m, d, "./visual/results/cbow.png")