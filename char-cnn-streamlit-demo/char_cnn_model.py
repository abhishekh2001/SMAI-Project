import torch.nn as nn
import torch

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=<>()[]{}"
char_map = {char:idx+1 for idx, char in enumerate(alphabet)}

def generate_embedding_matrix(size=len(alphabet)):
    return torch.vstack([torch.zeros(size), torch.eye(size)])

def preprocess_text(inp_string):
    inp_string = inp_string[:min(1014, len(inp_string))]
    inp_string = inp_string.lower()
    X_lis = [0] * 1014
    X_lis[:len(inp_string)] = [char_map.get(char, 0) for char in reversed(inp_string)]
    return torch.LongTensor(X_lis)

class CharCNN(nn.Module):
    def __init__(self, input_dim, enc_dim, output_dim, net_type="small"):
        super().__init__()
        
        self.input_dim = input_dim
        self.enc_dim = enc_dim
        
        if net_type == "small":
            self.latent_size = 1024
            self.cnn_features = 256
        else:
            self.latent_size = 2048
            self.cnn_features = 1024
        
        self.embed = nn.Embedding.from_pretrained(generate_embedding_matrix(), freeze=True)
        self.cnn, w_size = self.generate_cnn([
            # Kernel Size, Pooling Kernel Size
            [7, 3], # L1
            [7, 3], # L2
            [3, 0], # L3
            [3, 0], # L4
            [3, 0], # L5
            [3, 3]  # L6
        ])
        self.fc = nn.Sequential(
            nn.Linear(w_size * self.cnn_features, self.latent_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.latent_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.latent_size, output_dim)
        )
    
    def generate_cnn(self, layer_params):
        cnn_layers = []
        feature_w = self.input_dim
        for idx, (kernel_size, pool_size) in enumerate(layer_params):
            inp_size = self.enc_dim if idx == 0 else self.cnn_features
            feature_w = feature_w - kernel_size + 1
            cnn_layers.append(nn.Conv1d(inp_size, self.cnn_features, kernel_size))
            cnn_layers.append(nn.ReLU())
            if pool_size != 0:
                feature_w = feature_w // pool_size
                cnn_layers.append(nn.MaxPool1d(pool_size))
        return nn.Sequential(*cnn_layers), feature_w
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0, 0.05)
    
    def forward(self, x):
        embeddings = self.embed(x).permute(0, 2, 1)
        cnn_out = self.cnn(embeddings)
        fc_in = torch.flatten(cnn_out, start_dim=1)
        fc_out = self.fc(fc_in)
        return fc_out