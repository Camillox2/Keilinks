import requests
import threading
from bs4 import BeautifulSoup


class CrawlerBackground:
    def __init__(self, intervalo_minutos=5):
        self.intervalo_minutos = intervalo_minutos
        self._thread = None

    def iniciar(self):
        pass

    def crawl_agora(self, topicos=None):
        return 0


class ExtratorWeb:
    def extrair_texto(self, url):
        try:
            # Finge ser um browser normal para não ser bloqueado
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            resposta = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(resposta.text, 'html.parser')
            
            # Remove scripts e estilos
            for script in soup(["script", "style"]):
                script.extract()
                
            texto = soup.get_text(separator=' ')
            linhas = (line.strip() for line in texto.splitlines())
            chunks = (phrase.strip() for line in linhas for phrase in line.split("  "))
            texto_limpo = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Retorna no máximo 1000 caracteres para não estourar o limite da placa gráfica
            return texto_limpo[:1000] 
        except:
            return ""