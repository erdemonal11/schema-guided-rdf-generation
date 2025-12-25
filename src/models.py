import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, embedding_dim=256, hidden_dim=512, num_relations=0):
        super(Generator, self).__init__()
        self.rel_embedding = nn.Embedding(num_relations, embedding_dim)

        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, noise, relation_ids):
        r_emb = self.rel_embedding(relation_ids)
        x = torch.cat([noise, r_emb], dim=1)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=256, hidden_dim=512):
        super(Discriminator, self).__init__()
        self.ent_embedding = nn.Embedding(num_entities, embedding_dim)
        self.rel_embedding = nn.Embedding(num_relations, embedding_dim)

        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, head_ids, rel_ids, tail_embedding):
        h_emb = self.ent_embedding(head_ids)
        r_emb = self.rel_embedding(rel_ids)
        x = torch.cat([h_emb, r_emb, tail_embedding], dim=1)
        return self.net(x)

    def get_entity_embedding(self, entity_ids):
        return self.ent_embedding(entity_ids)