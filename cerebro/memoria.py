"""
Memória de Longo Prazo da Keilinks
Lembra de contexto entre conversas:
- Humor do Vitor
- Temas recorrentes
- Coisas que ele disse
"""

import os
import json
from datetime import datetime, timedelta


class Memoria:
    def __init__(self, caminho: str):
        self.caminho = caminho
        self.dados = {
            'humor_atual': 'neutro',
            'ultima_interacao': None,
            'temas_recentes': [],
            'fatos_sobre_usuario': [],
            'notas': [],
        }
        self.carregar()

    def carregar(self):
        if os.path.exists(self.caminho):
            with open(self.caminho, 'r', encoding='utf-8') as f:
                try:
                    self.dados = json.load(f)
                except json.JSONDecodeError:
                    pass
            print(f"[Memoria] Carregada ({len(self.dados.get('fatos_sobre_usuario', []))} fatos do usuario)")

    def salvar(self):
        os.makedirs(os.path.dirname(self.caminho), exist_ok=True)
        self.dados['ultima_interacao'] = datetime.now().isoformat()
        with open(self.caminho, 'w', encoding='utf-8') as f:
            json.dump(self.dados, f, ensure_ascii=False, indent=2)

    def detectar_humor(self, mensagem: str) -> str:
        """Detecta humor pela mensagem"""
        msg = mensagem.lower()

        indicadores = {
            'feliz':   ['feliz', 'to feliz', 'animado', 'boa', 'legal', 'show', 'incrivel', 'top', 'demais', 'kkk', 'haha'],
            'triste':  ['triste', 'to triste', 'mal', 'chateado', 'desanimado', 'chorando', 'sozinho', 'saudade'],
            'raiva':   ['raiva', 'puto', 'irritado', 'odeio', 'merda', 'porra', 'caralho', 'desgraça'],
            'ansioso': ['ansioso', 'nervoso', 'preocupado', 'medo', 'tenso'],
            'cansado': ['cansado', 'exausto', 'sono', 'esgotado', 'morto'],
            'entediado': ['entediado', 'tedio', 'nada pra fazer', 'chato'],
        }

        for humor, palavras in indicadores.items():
            if any(p in msg for p in palavras):
                return humor
        return 'neutro'

    def atualizar(self, mensagem: str, resposta: str):
        """Atualiza memória com base na conversa"""
        humor = self.detectar_humor(mensagem)
        if humor != 'neutro':
            self.dados['humor_atual'] = humor

        # Extrai se o usuário mencionou algo sobre si mesmo
        msg = mensagem.lower()
        gatilhos_pessoais = [
            'eu gosto', 'eu amo', 'eu odeio', 'eu prefiro',
            'meu favorito', 'minha favorita', 'eu trabalho',
            'eu estudo', 'eu moro', 'eu tenho',
        ]
        for g in gatilhos_pessoais:
            if g in msg:
                fato = mensagem.strip()
                if fato not in self.dados.get('fatos_sobre_usuario', []):
                    self.dados.setdefault('fatos_sobre_usuario', []).append(fato)
                break

        # Temas recentes (últimos 10)
        temas = self.dados.setdefault('temas_recentes', [])
        temas.append({'tema': mensagem[:80], 'data': datetime.now().isoformat()})
        self.dados['temas_recentes'] = temas[-10:]

        self.salvar()

    def gerar_contexto(self) -> str:
        """Gera contexto extra baseado na memória para injetar nas respostas"""
        partes = []

        # Tempo desde última interação
        ultima = self.dados.get('ultima_interacao')
        if ultima:
            try:
                dt = datetime.fromisoformat(ultima)
                diff = datetime.now() - dt
                if diff > timedelta(hours=6):
                    partes.append(f"Faz {diff.seconds//3600} horas que não conversamos.")
                if diff > timedelta(days=1):
                    partes.append(f"Faz {diff.days} dia(s) que não conversamos.")
            except (ValueError, TypeError):
                pass

        # Humor
        humor = self.dados.get('humor_atual', 'neutro')
        if humor != 'neutro':
            partes.append(f"Ultimo humor detectado: {humor}.")

        return ' '.join(partes) if partes else ''

    def get_saudacao(self) -> str | None:
        """Gera saudação contextual baseada na memória"""
        ultima = self.dados.get('ultima_interacao')
        humor = self.dados.get('humor_atual', 'neutro')

        if not ultima:
            return None

        try:
            dt = datetime.fromisoformat(ultima)
            diff = datetime.now() - dt

            if diff > timedelta(days=1):
                if humor == 'triste':
                    return "Faz um tempo que você não aparecia. Ta melhor?"
                return None

            if humor == 'triste':
                return "E ai, ta melhor desde a ultima vez?"
            if humor == 'raiva':
                return "Esfriou a cabeca?"
        except (ValueError, TypeError):
            pass

        return None
