import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.dataset import DBLPDataset
from src.models import Generator, Discriminator

DATA_PATH = Path("data/processed/kg_triples_ids.txt")
SYNTHETIC_DIR = Path("data/synthetic")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "gan_latest.pth"
LOG_FILE = Path("data/processed/training_log.csv")

EMBEDDING_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 1024
MAX_EPOCHS = 1000
EPOCHS_PER_RUN = 2
LR = 0.0002
CLIP_VALUE = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    if not DATA_PATH.exists():
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        return

    dataset = DBLPDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator(EMBEDDING_DIM, HIDDEN_DIM, dataset.num_relations).to(device)
    D = Discriminator(dataset.num_entities, dataset.num_relations, EMBEDDING_DIM, HIDDEN_DIM).to(device)

    optimizer_G = optim.RMSprop(G.parameters(), lr=LR)
    optimizer_D = optim.RMSprop(D.parameters(), lr=LR)

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    start_epoch = 0

    if CHECKPOINT_PATH.exists():
        print(f"[INFO] Loading previous brain from: {CHECKPOINT_PATH}")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            G.load_state_dict(checkpoint["G_state"])
            D.load_state_dict(checkpoint["D_state"])
            start_epoch = int(checkpoint.get("epoch", 0))
            print(f"[SUCCESS] Checkpoint loaded. Resuming training from epoch {start_epoch}.")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint: {e}")
            print("[INFO] Starting fresh training.")
    else:
        print("[INFO] No checkpoint found. Starting fresh training.")

    if not LOG_FILE.exists():
        with open(LOG_FILE, "w") as f:
            f.write("Epoch,D_Loss,G_Loss\n")

    if start_epoch >= MAX_EPOCHS:
        print(f"[INFO] MAX_EPOCHS={MAX_EPOCHS} already reached. Skipping further training.")
    else:
        end_epoch = min(start_epoch + EPOCHS_PER_RUN, MAX_EPOCHS)
        print(f"[INFO] Starting WGAN training on {device} from epoch {start_epoch+1} to {end_epoch}...")

        for epoch in range(start_epoch, end_epoch):
            total_d_loss = 0
            total_g_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False)
            
            for i, batch in enumerate(pbar):
                real_h = batch['head'].to(device)
                real_r = batch['relation'].to(device)
                real_t = batch['tail'].to(device)
                batch_len = real_h.size(0)

                optimizer_D.zero_grad()

                real_t_emb = D.get_entity_embedding(real_t)
                d_real = D(real_h, real_r, real_t_emb).mean()

                noise = torch.randn(batch_len, EMBEDDING_DIM).to(device)
                fake_t_emb = G(noise, real_r).detach()
                d_fake = D(real_h, real_r, fake_t_emb).mean()

                d_loss = -(d_real - d_fake)
                d_loss.backward()
                optimizer_D.step()

                for p in D.parameters():
                    p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

                total_d_loss += d_loss.item()

                if i % 5 == 0:
                    optimizer_G.zero_grad()
                    noise = torch.randn(batch_len, EMBEDDING_DIM).to(device)
                    fake_t_emb = G(noise, real_r)
                    g_loss = -D(real_h, real_r, fake_t_emb).mean()
                    g_loss.backward()
                    optimizer_G.step()
                    total_g_loss += g_loss.item()
                
                pbar.set_postfix({'D_Loss': f"{d_loss.item():.4f}"})

            avg_d_loss = total_d_loss / len(dataloader)
            avg_g_loss = total_g_loss / len(dataloader)
            
            with open(LOG_FILE, "a") as f:
                f.write(f"{epoch+1},{avg_d_loss:.6f},{avg_g_loss:.6f}\n")

    print(f"[FINISHED] Training step complete.")
    
    G.eval()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_file = SYNTHETIC_DIR / f"generated_{timestamp}.txt"
    
    type_indices = {
        "pub": [i for i, e in enumerate(dataset.entity_list) if e.startswith("pub_")],
        "author": [i for i, e in enumerate(dataset.entity_list) if e.startswith("author_")],
        "venue": [i for i, e in enumerate(dataset.entity_list) if e.startswith("venue_")],
        "year": [i for i, e in enumerate(dataset.entity_list) if e.startswith("year_")]
    }
    
    relation_tail_rules = {
        "dblp:wrote": "pub",
        "dblp:hasAuthor": "author",
        "dblp:publishedIn": "venue",
        "dblp:inYear": "year"
    }

    relation_head_rules = {
        "dblp:wrote": "author",
        "dblp:hasAuthor": "pub",
        "dblp:publishedIn": "pub",
        "dblp:inYear": "pub"
    }
    
    all_ent_emb = D.ent_embedding.weight.data
    
    with torch.no_grad():
        num_samples = 1000
        test_r = torch.randint(0, dataset.num_relations, (num_samples,)).to(device)
        noise = torch.randn(num_samples, EMBEDDING_DIM).to(device)
        
        fake_emb = G(noise, test_r)
        
        with open(output_file, "w") as f:
            f.write("HEAD_ID\tRELATION_ID\tGENERATED_TAIL_ID\tDISTANCE_SCORE\n")
            
            for k in range(num_samples):
                r_str = dataset.relation_list[test_r[k].item()]

                head_type = relation_head_rules.get(r_str, "author")
                head_candidates = type_indices.get(head_type, list(range(len(dataset.entity_list))))
                head_candidates_tensor = torch.tensor(head_candidates, device=device)
                head_choice = torch.randint(0, len(head_candidates_tensor), (1,)).item()
                h_idx = head_candidates[head_choice]
                h_str = dataset.entity_list[h_idx]

                tail_type = relation_tail_rules.get(r_str, "pub")
                tail_candidates = type_indices.get(tail_type, list(range(len(dataset.entity_list))))
                tail_candidates_tensor = torch.tensor(tail_candidates, device=device)
                candidate_embeddings = all_ent_emb[tail_candidates_tensor]
                
                current_emb = fake_emb[k].unsqueeze(0)
                dist = torch.norm(candidate_embeddings - current_emb, dim=1)

                K = min(50, len(dist))  
                top_dists, top_indices = torch.topk(dist, k=K, largest=False)

                min_skip = min(5, K - 1)
                rand_choice = torch.randint(min_skip, K, (1,)).item()

                best_local_idx = top_indices[rand_choice].item()
                best_global_idx = tail_candidates[best_local_idx]

                t_str = dataset.entity_list[best_global_idx]
                score = top_dists[rand_choice].item()
                f.write(f"{h_str}\t{r_str}\t{t_str}\t{score:.4f}\n")

    torch.save(
        {
            "G_state": G.state_dict(),
            "D_state": D.state_dict(),
            "epoch": min(MAX_EPOCHS, end_epoch if "end_epoch" in locals() else start_epoch),
        },
        CHECKPOINT_PATH,
    )
    print("[SUCCESS] Checkpoint saved.")

if __name__ == "__main__":
    train()