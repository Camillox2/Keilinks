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


class Memoria:
    def __init__(self, caminho='dados/memoria.json'):
        self.caminho = caminho
        self.dados = {}
        self._usuarios = {}
        self._carregar()

    def _carregar(self):
        if os.path.exists(self.caminho):
            try:
                with open(self.caminho, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    self.dados = obj.get('global', obj)
                    self._usuarios = obj.get('usuarios', {})
            except Exception:
                pass

    def _salvar(self):
        os.makedirs(os.path.dirname(self.caminho) or '.', exist_ok=True)
        with open(self.caminho, 'w', encoding='utf-8') as f:
            json.dump({'global': self.dados, 'usuarios': self._usuarios}, f, ensure_ascii=False, indent=2)

    def gerar_contexto(self, user_id=None):
        partes = []
        if user_id and str(user_id) in self._usuarios:
            u = self._usuarios[str(user_id)]
            if u.get('nome'):
                partes.append(f"usuario: {u['nome']}")
            if u.get('interesses'):
                partes.append(f"interesses: {', '.join(u['interesses'][-3:])}")
        if self.dados.get('humor_atual'):
            partes.append(f"humor: {self.dados['humor_atual']}")
        return ' | '.join(partes)

    def atualizar(self, pergunta, resposta, user_id=None):
        if user_id:
            uid = str(user_id)
            if uid not in self._usuarios:
                self._usuarios[uid] = {'interesses': []}
            palavras = [w for w in pergunta.lower().split() if len(w) >= 4]
            if palavras:
                self._usuarios[uid]['interesses'].extend(palavras[:3])
                self._usuarios[uid]['interesses'] = self._usuarios[uid]['interesses'][-20:]
        try:
            self._salvar()
        except Exception:
            pass