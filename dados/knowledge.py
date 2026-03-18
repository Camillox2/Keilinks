"""
Knowledge Storage da Keilinks v5
Agora usa MySQL ao inves de JSON.
Busca textual via FULLTEXT index.
"""

from dados.database import knowledge_adicionar, knowledge_buscar, knowledge_total, knowledge_por_fonte


class Knowledge:
    def __init__(self, caminho: str = None):
        # caminho mantido para compatibilidade, mas nao usado
        self._caminho = caminho
        print(f"[Knowledge] MySQL conectado ({self.total()} fatos)")

    def carregar(self):
        """Noop — MySQL e sempre atualizado"""
        pass

    def salvar(self):
        """Noop — MySQL persiste automaticamente"""
        pass

    def adicionar(self, pergunta: str, resposta: str, fonte: str = 'conversa'):
        knowledge_adicionar(pergunta, resposta, fonte)

    def buscar(self, pergunta: str) -> str | None:
        resultados = knowledge_buscar(pergunta, limite=1)
        if resultados:
            return resultados[0]['resposta']
        return None

    def total(self) -> int:
        return knowledge_total()

    def por_fonte(self) -> dict:
        return knowledge_por_fonte()
