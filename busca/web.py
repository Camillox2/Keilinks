try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

class BuscadorWeb:
    def __init__(self):
        self.ddgs = DDGS()

    def pesquisar(self, query, max_resultados=3):
        try:
            # Faz a pesquisa silenciosamente
            resultados = self.ddgs.text(query, max_results=max_resultados)
            if not resultados:
                return ""

            contexto = []
            for r in resultados:
                contexto.append(f"Título: {r['title']} | Resumo: {r['body']}")

            return " ".join(contexto)
        except Exception as e:
            return ""


_buscador = BuscadorWeb()

def pesquisar(query, max_resultados=3):
    return _buscador.pesquisar(query, max_resultados)

def precisa_buscar(texto):
    palavras_chave = ['quem', 'o que', 'quando', 'onde', 'como', 'por que', 'qual',
                      'noticia', 'hoje', 'agora', 'atual', 'ultimo', 'recente']
    t = texto.lower()
    return any(p in t for p in palavras_chave)