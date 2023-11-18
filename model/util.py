import torch

def cos_similarity(A, B):
    return torch.dot(A, B)/(torch.norm(A)*torch.norm(B))