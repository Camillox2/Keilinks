"""
Memoria de Longo Prazo da Keilinks
Lembra de contexto entre conversas:
- Humor do usuario
- Temas recorrentes
- Coisas que ele disse
- Suporte a memoria por usuario (cada um tem seus fatos)
"""

import os
import json
from datetime import datetime, timedelta


def _dados_padrao():
    return {
        'humor_atual': 'neutro',
        'ultima_interacao': None,
        'temas_recentes': [],
        'fatos_sobre_usuario': [],
        'notas': [],
    }


class Memoria:
    def __init__(self, caminho: str):
        self.caminho = caminho
        self.dados = _dados_padrao()
        self._usuarios = {}  # {user_id: dados}
        self.carregar()

    def carregar(self):
        if os.path.exists(self.caminho):
            with open(self.caminho, 'r', encoding='utf-8') as f:
                try:
                    raw = json.load(f)
                    if '_usuarios' in raw:
                        self._usuarios = raw['_usuarios']
                        self.dados = raw.get('_global', _dados_padrao())
                    else:
                        self.dados = raw
                except json.JSONDecodeError:
                    pass
            n_fatos = len(self.dados.get('fatos_sobre_usuario', []))
            n_users = len(self._usuarios)
            print(f"[Memoria] Carregada ({n_fatos} fatos globais, {n_users} usuarios)")

    def salvar(self):
        os.makedirs(os.path.dirname(self.caminho), exist_ok=True)
        self.dados['ultima_interacao'] = datetime.now().isoformat()
        payload = {
            '_global': self.dados,
            '_usuarios': self._usuarios,
        }
        with open(self.caminho, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _get_user_data(self, user_id):
        uid = str(user_id)
        if uid not in self._usuarios:
            self._usuarios[uid] = _dados_padrao()
        return self._usuarios[uid]

    def detectar_humor(self, mensagem: str) -> str:
        msg = mensagem.lower()
        indicadores = {
            'feliz':   ['feliz', 'to feliz', 'animado', 'boa', 'legal', 'show', 'incrivel', 'top', 'demais', 'kkk', 'haha'],
            'triste':  ['triste', 'to triste', 'mal', 'chateado', 'desanimado', 'chorando', 'sozinho', 'saudade'],
            'raiva':   ['raiva', 'puto', 'irritado', 'odeio', 'merda', 'porra', 'caralho', 'desgraca'],
            'ansioso': ['ansioso', 'nervoso', 'preocupado', 'medo', 'tenso'],
            'cansado': ['cansado', 'exausto', 'sono', 'esgotado', 'morto'],
            'entediado': ['entediado', 'tedio', 'nada pra fazer', 'chato'],
        }
        for humor, palavras in indicadores.items():
            if any(p in msg for p in palavras):
                return humor
        return 'neutro'

    def atualizar(self, mensagem: str, resposta: str, user_id=None):
        dados = self._get_user_data(user_id) if user_id else self.dados
        humor = self.detectar_humor(mensagem)
        if humor != 'neutro':
            dados['humor_atual'] = humor
        msg = mensagem.lower()
        gatilhos_pessoais = [
            'eu gosto', 'eu amo', 'eu odeio', 'eu prefiro',
            'meu favorito', 'minha favorita', 'eu trabalho',
            'eu estudo', 'eu moro', 'eu tenho',
            'meu nome', 'me chamo', 'eu sou',
        ]
        for g in gatilhos_pessoais:
            if g in msg:
                fato = mensagem.strip()
                fatos = dados.setdefault('fatos_sobre_usuario', [])
                if fato not in fatos and len(fatos) < 50:
                    fatos.append(fato)
                break
        temas = dados.setdefault('temas_recentes', [])
        temas.append({'tema': mensagem[:80], 'data': datetime.now().isoformat()})
        dados['temas_recentes'] = temas[-10:]
        self.salvar()

    def gerar_contexto(self, user_id=None) -> str:
        dados = self._get_user_data(user_id) if user_id else self.dados
        partes = []
        ultima = dados.get('ultima_interacao')
        if ultima:
            try:
                dt = datetime.fromisoformat(ultima)
                diff = datetime.now() - dt
                if diff > timedelta(days=1):
                    partes.append(f"faz {diff.days} dia(s) que nao conversamos")
                elif diff > timedelta(hours=6):
                    h = diff.seconds // 3600
                    partes.append(f"faz {h} horas que nao conversamos")
            except (ValueError, TypeError):
                pass
        humor = dados.get('humor_atual', 'neutro')
        if humor != 'neutro':
            partes.append(f"usuario ta {humor}")
        fatos = dados.get('fatos_sobre_usuario', [])
        if fatos:
            ultimos = fatos[-5:]
            partes.append(f"sei sobre ele: {'; '.join(ultimos)}")
        temas = dados.get('temas_recentes', [])
        if temas:
            temas_txt = [t['tema'] for t in temas[-3:]]
            partes.append(f"temas recentes: {', '.join(temas_txt)}")
        return '. '.join(partes) if partes else ''

    def get_saudacao(self, user_id=None):
        dados = self._get_user_data(user_id) if user_id else self.dados
        ultima = dados.get('ultima_interacao')
        humor = dados.get('humor_atual', 'neutro')
        if not ultima:
            return None
        try:
            dt = datetime.fromisoformat(ultima)
            diff = datetime.now() - dt
            if diff > timedelta(days=1):
                if humor == 'triste':
                    return "Faz um tempo que voce nao aparecia. Ta melhor?"
                return None
            if humor == 'triste':
                return "E ai, ta melhor desde a ultima vez?"
            if humor == 'raiva':
                return "Esfriou a cabeca?"
        except (ValueError, TypeError):
            pass
        return None
