import torch
from sentence_transformers import SentenceTransformer

class GeradorEmbeddings:
    def __init__(self, modelo_nome='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.modelo = SentenceTransformer(modelo_nome).to(self.device)

    def gerar_vetor(self, texto):
        vetor = self.modelo.encode(texto, convert_to_tensor=True, device=self.device)
        return vetor.cpu().tolist()