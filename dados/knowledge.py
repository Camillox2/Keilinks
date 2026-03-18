"""
Knowledge Storage da Keilinks v6
MySQL + busca semantica por embeddings
FULLTEXT como fallback se embeddings indisponivel
"""

from dados.database import knowledge_adicionar, knowledge_buscar, knowledge_total, knowledge_por_fonte

# Busca semantica no knowledge (lazy init)
_indice_knowledge = None
_knowledge_ok = False


def _iniciar_knowledge_embeddings():
    """Constroi indice semantico do knowledge (chamado 1x)"""
    global _indice_knowledge, _knowledge_ok
    if _indice_knowledge is not None:
        return

    try:
        from cerebro.embeddings import IndiceSemantico
        from dados.database import get_conn
        import os

        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT pergunta, resposta FROM knowledge ORDER BY id")
            rows = cur.fetchall()
        conn.close()

        if not rows:
            _knowledge_ok = False
            return

        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knowledge_cache.npy')
        _indice_knowledge = IndiceSemantico(cache_path=cache)

        perguntas = [r['pergunta'] for r in rows]
        respostas = [r['resposta'] for r in rows]
        _indice_knowledge.construir(perguntas, respostas)
        _knowledge_ok = True
    except Exception as e:
        print(f"[Knowledge] Embeddings indisponivel: {e}")
        _knowledge_ok = False


class Knowledge:
    def __init__(self, caminho: str = None):
        self._caminho = caminho
        print(f"[Knowledge] MySQL conectado ({self.total()} fatos)")

    def carregar(self):
        """Noop — MySQL e sempre atualizado"""
        pass

    def salvar(self):
        """Noop — MySQL persiste automaticamente"""
        pass

    def iniciar_embeddings(self):
        """Inicializa busca semantica (chamar apos startup)"""
        _iniciar_knowledge_embeddings()

    def adicionar(self, pergunta: str, resposta: str, fonte: str = 'conversa'):
        knowledge_adicionar(pergunta, resposta, fonte)
        # Atualiza indice semantico
        global _indice_knowledge, _knowledge_ok
        if _knowledge_ok and _indice_knowledge:
            try:
                _indice_knowledge.adicionar(pergunta, resposta)
            except Exception:
                pass

    def buscar(self, pergunta: str) -> str | None:
        """Busca semantica com fallback para FULLTEXT"""
        # Tenta semantica primeiro
        global _knowledge_ok, _indice_knowledge
        if _knowledge_ok and _indice_knowledge:
            try:
                resultados = _indice_knowledge.buscar(pergunta, top_k=1)
                if resultados:
                    _, resposta, score = resultados[0]
                    if score >= 0.35:
                        return resposta
            except Exception:
                pass

        # Fallback: FULLTEXT MySQL
        resultados = knowledge_buscar(pergunta, limite=1)
        if resultados:
            return resultados[0]['resposta']
        return None

    def total(self) -> int:
        return knowledge_total()

    def por_fonte(self) -> dict:
        return knowledge_por_fonte()
