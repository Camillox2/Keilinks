import os
import json
import torch
import torch.nn.functional as F
from cerebro.embeddings import GeradorEmbeddings

class MemoriaLongoPrazo:
    def __init__(self, caminho_banco='dados/memoria.json'):
        self.caminho_banco = caminho_banco
        self.gerador = GeradorEmbeddings()
        self.memorias = []
        self.vetores = None
        self.carregar_banco()

    def carregar_banco(self):
        if os.path.exists(self.caminho_banco):
            with open(self.caminho_banco, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                self.memorias = dados.get('memorias', [])
                vetores_lista = dados.get('vetores', [])
                if self.memorias and vetores_lista:
                    self.vetores = torch.tensor(vetores_lista)
        else:
            self.memorias = []
            self.vetores = None

    def salvar_banco(self):
        os.makedirs(os.path.dirname(self.caminho_banco), exist_ok=True)
        dados = {
            'memorias': self.memorias,
            'vetores': self.vetores.tolist() if self.vetores is not None else []
        }
        with open(self.caminho_banco, 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False)

    def adicionar_memoria(self, texto):
        vetor = self.gerador.gerar_vetor(texto)
        vetor_tensor = torch.tensor([vetor])
        
        self.memorias.append(texto)
        
        if self.vetores is None:
            self.vetores = vetor_tensor
        else:
            self.vetores = torch.cat((self.vetores, vetor_tensor), dim=0)
            
        self.salvar_banco()

    def buscar_memoria(self, pergunta, top_k=2, limite_similaridade=0.5):
        if self.vetores is None or len(self.memorias) == 0:
            return []
            
        vetor_pergunta = torch.tensor([self.gerador.gerar_vetor(pergunta)])
        
        similaridades = F.cosine_similarity(vetor_pergunta, self.vetores)
        
        valores, indices = torch.topk(similaridades, min(top_k, len(self.memorias)))
        
        resultados = []
        for val, idx in zip(valores, indices):
            if val.item() >= limite_similaridade:
                resultados.append(self.memorias[idx.item()])
                
        return resultados