import torch
import torch.nn as nn

'''
Music Transformer nn class
'''
class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, beat_vocab_size, vel_vocab_size, embed_dim=256, max_seq_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.beat_embedding = nn.Embedding(beat_vocab_size, embed_dim)
        self.vel_embedding = nn.Embedding(vel_vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.out = nn.Linear(embed_dim, vocab_size)
        self.beat_out = nn.Linear(embed_dim, beat_vocab_size)
        self.vel_out = nn.Linear(embed_dim, vel_vocab_size)
        
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x[:, :, 0]) + self.beat_embedding(x[:, :, 1]) + self.vel_embedding(x[:, :, 2]) + self.pos_embedding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.out(x), self.beat_out(x), self.vel_out(x)
    
    def generate(self, seed, temp=0.8):    
        was_training = self.training
        self.eval()
        
        seq_length = seed.shape[1]
        new_seq = seed.detach().clone()
        with torch.no_grad():
            for _ in range(seq_length):
                pitch_out, beat_out, vel_out = self(new_seq)
                
                pitch_logits = torch.softmax(pitch_out[0, -1, :] / temp, -1)
                beat_logits = torch.softmax(beat_out[0, -1, :] / temp, -1)
                vel_logits = torch.softmax(vel_out[0, -1, :] / temp, -1)
                
                sample_pitch = torch.multinomial(pitch_logits, 1).view(1, 1, 1)
                sample_beat = torch.multinomial(beat_logits, 1).view(1, 1, 1)
                sample_vel = torch.multinomial(vel_logits, 1).view(1, 1, 1)

                combined_sample = torch.cat([sample_pitch, sample_beat, sample_vel], dim=2)
                
                new_seq = torch.cat([new_seq, combined_sample], dim=1)[:, 1:]
        
        self.train(was_training)
        return new_seq