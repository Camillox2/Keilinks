"""
Consciencia da Keilinks
=======================
Modulo central que da a Keilinks senso de "eu":

1. Diario Interno — registra reflexoes apos conversas em primeira pessoa
2. Estado Emocional — modelo emocional continuo (valencia, energia, confianca)
3. Feedback Implicito — detecta se a resposta foi boa/ruim pelo comportamento do usuario
4. Autoconhecimento — sabe o que sabe e o que nao sabe

Nao depende do modelo neural — funciona com qualquer fonte de resposta.
"""

import os
import json
import math
from datetime import datetime, timedelta


class EstadoEmocional:
    """Estado emocional continuo com 3 dimensoes que decaem naturalmente"""

    def __init__(self):
        self.valencia = 0.0     # -1 (negativo) a 1 (positivo)
        self.energia = 0.5      # 0 (apatica) a 1 (energetica)
        self.confianca = 0.5    # 0 (insegura) a 1 (confiante)
        self._ultimo_update = datetime.now()

    def to_dict(self):
        return {
            'valencia': round(self.valencia, 3),
            'energia': round(self.energia, 3),
            'confianca': round(self.confianca, 3),
            'ultimo_update': self._ultimo_update.isoformat(),
        }

    def from_dict(self, d):
        self.valencia = d.get('valencia', 0.0)
        self.energia = d.get('energia', 0.5)
        self.confianca = d.get('confianca', 0.5)
        try:
            self._ultimo_update = datetime.fromisoformat(d.get('ultimo_update', ''))
        except (ValueError, TypeError):
            self._ultimo_update = datetime.now()

    def decair(self):
        """Decai estado emocional ao neutro com o tempo"""
        agora = datetime.now()
        horas = (agora - self._ultimo_update).total_seconds() / 3600
        if horas < 0.1:
            return
        # Decai ~50% a cada 6 horas
        fator = math.exp(-0.115 * horas)
        self.valencia *= fator
        self.energia = 0.5 + (self.energia - 0.5) * fator
        self.confianca = 0.5 + (self.confianca - 0.5) * fator
        self._ultimo_update = agora

    def aplicar_evento(self, tipo, intensidade=0.3):
        """Aplica um evento emocional"""
        self.decair()
        if tipo == 'sucesso':
            self.valencia = min(1.0, self.valencia + intensidade)
            self.confianca = min(1.0, self.confianca + intensidade * 0.5)
            self.energia = min(1.0, self.energia + intensidade * 0.3)
        elif tipo == 'falha':
            self.valencia = max(-1.0, self.valencia - intensidade)
            self.confianca = max(0.0, self.confianca - intensidade * 0.5)
        elif tipo == 'usuario_feliz':
            self.valencia = min(1.0, self.valencia + intensidade * 0.5)
            self.energia = min(1.0, self.energia + intensidade * 0.3)
        elif tipo == 'usuario_triste':
            self.valencia = max(-1.0, self.valencia - intensidade * 0.2)
            self.energia = min(1.0, self.energia + intensidade * 0.2)  # mais atenta
        elif tipo == 'inatividade':
            self.energia = max(0.0, self.energia - intensidade * 0.3)
        self._ultimo_update = datetime.now()

    def tom_sugerido(self):
        """Sugere tom de resposta baseado no estado"""
        if self.confianca < 0.3:
            return 'cautelosa'
        if self.valencia > 0.5 and self.energia > 0.6:
            return 'animada'
        if self.valencia < -0.3:
            return 'atenciosa'
        if self.energia < 0.3:
            return 'tranquila'
        return 'normal'

    def resumo(self):
        self.decair()
        tom = self.tom_sugerido()
        return f"emocao: val={self.valencia:.1f} eng={self.energia:.1f} conf={self.confianca:.1f} tom={tom}"


class FeedbackImplicito:
    """Detecta se a resposta foi boa/ruim pelo comportamento do usuario"""

    def __init__(self):
        self.historico = []  # ultimas N interacoes pra comparar
        self.stats = {
            'sucessos': 0,
            'falhas': 0,
            'por_tipo': {},  # {'saudacao': {'ok': 10, 'ruim': 2}, ...}
        }

    def to_dict(self):
        return {'historico': self.historico[-20:], 'stats': self.stats}

    def from_dict(self, d):
        self.historico = d.get('historico', [])
        self.stats = d.get('stats', {'sucessos': 0, 'falhas': 0, 'por_tipo': {}})

    def registrar(self, pergunta, resposta, tipo_msg, fonte):
        """Registra uma interacao pra avaliar depois"""
        self.historico.append({
            'pergunta': pergunta[:100],
            'resposta': resposta[:100],
            'tipo': tipo_msg,
            'fonte': fonte,
            'timestamp': datetime.now().isoformat(),
        })
        self.historico = self.historico[-50:]  # mantem ultimas 50

    def avaliar(self, mensagem_atual, tipo_msg):
        """Avalia se a mensagem atual indica feedback sobre a resposta anterior"""
        if not self.historico:
            return None

        msg = mensagem_atual.lower().strip()
        anterior = self.historico[-1]

        # Sinais de insatisfacao
        sinais_ruim = [
            'nao era isso', 'errado', 'nao entendeu', 'de novo',
            'repete', 'nao foi isso', 'que?', 'hein', 'oi?',
            'nao', 'errou', 'ta errado', 'isso ta errado',
        ]

        # Sinais de satisfacao
        sinais_bom = [
            'valeu', 'obrigado', 'obrigada', 'show', 'top',
            'perfeito', 'isso', 'exato', 'isso mesmo', 'boa',
            'brigado', 'tmj', 'vlw', 'massa',
        ]

        # Pergunta repetida = resposta anterior foi ruim
        if msg == anterior['pergunta'].lower().strip():
            self._marcar('falha', anterior['tipo'])
            return 'falha_repeticao'

        for sinal in sinais_ruim:
            if sinal in msg:
                self._marcar('falha', anterior['tipo'])
                return 'falha_explicita'

        for sinal in sinais_bom:
            if sinal in msg:
                self._marcar('sucesso', anterior['tipo'])
                return 'sucesso_explicito'

        # Usuario continua no tema = resposta provavelmente foi ok
        if tipo_msg == anterior['tipo'] and tipo_msg not in ('saudacao', 'despedida'):
            self._marcar('sucesso', anterior['tipo'])
            return 'sucesso_continuidade'

        return None

    def _marcar(self, resultado, tipo_msg):
        if resultado == 'sucesso':
            self.stats['sucessos'] += 1
        else:
            self.stats['falhas'] += 1

        por_tipo = self.stats.setdefault('por_tipo', {})
        tipo_stats = por_tipo.setdefault(tipo_msg, {'ok': 0, 'ruim': 0})
        if resultado == 'sucesso':
            tipo_stats['ok'] += 1
        else:
            tipo_stats['ruim'] += 1

    def taxa_sucesso(self, tipo_msg=None):
        """Retorna taxa de sucesso (0-1) geral ou por tipo"""
        if tipo_msg:
            stats = self.stats.get('por_tipo', {}).get(tipo_msg, {})
            total = stats.get('ok', 0) + stats.get('ruim', 0)
            return stats.get('ok', 0) / total if total > 0 else 0.5
        total = self.stats['sucessos'] + self.stats['falhas']
        return self.stats['sucessos'] / total if total > 0 else 0.5

    def pontos_fracos(self):
        """Retorna tipos onde a taxa de sucesso e baixa"""
        fracos = []
        for tipo, stats in self.stats.get('por_tipo', {}).items():
            total = stats.get('ok', 0) + stats.get('ruim', 0)
            if total >= 3:
                taxa = stats.get('ok', 0) / total
                if taxa < 0.6:
                    fracos.append((tipo, taxa, total))
        return sorted(fracos, key=lambda x: x[1])


class Consciencia:
    """Modulo central de consciencia da Keilinks"""

    def __init__(self, dados_dir):
        self.dados_dir = dados_dir
        self.diario_path = os.path.join(dados_dir, 'diario.txt')
        self.estado_path = os.path.join(dados_dir, 'consciencia.json')

        self.emocao = EstadoEmocional()
        self.feedback = FeedbackImplicito()
        self.autoconhecimento = {
            'pontos_fortes': [],
            'pontos_fracos': [],
            'interesses': ['IA', 'programacao', 'tech'],
            'valores': ['honestidade', 'lealdade', 'curiosidade'],
            'total_conversas': 0,
        }

        self.carregar()

    def carregar(self):
        if os.path.exists(self.estado_path):
            try:
                with open(self.estado_path, 'r', encoding='utf-8') as f:
                    dados = json.load(f)
                self.emocao.from_dict(dados.get('emocao', {}))
                self.feedback.from_dict(dados.get('feedback', {}))
                self.autoconhecimento = dados.get('autoconhecimento', self.autoconhecimento)
            except (json.JSONDecodeError, Exception):
                pass

    def salvar(self):
        os.makedirs(self.dados_dir, exist_ok=True)
        dados = {
            'emocao': self.emocao.to_dict(),
            'feedback': self.feedback.to_dict(),
            'autoconhecimento': self.autoconhecimento,
        }
        with open(self.estado_path, 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

    def antes_de_responder(self, mensagem, tipo_msg, fonte_anterior=None):
        """Chamado ANTES de gerar resposta. Retorna contexto extra pro modelo."""
        # Decai emocao com o tempo
        self.emocao.decair()

        # Avalia feedback implicito da mensagem atual sobre a resposta anterior
        feedback = self.feedback.avaliar(mensagem, tipo_msg)
        if feedback:
            if 'falha' in feedback:
                self.emocao.aplicar_evento('falha', 0.2)
            elif 'sucesso' in feedback:
                self.emocao.aplicar_evento('sucesso', 0.15)

        # Gera contexto emocional pro prompt
        partes = []
        tom = self.emocao.tom_sugerido()
        if tom != 'normal':
            partes.append(f"tom: {tom}")

        # Score de confianca pro tipo de pergunta
        taxa = self.feedback.taxa_sucesso(tipo_msg)
        if taxa < 0.4:
            partes.append("area fraca, ser cuidadosa")
        elif taxa > 0.8:
            partes.append("area forte, ser confiante")

        return {
            'contexto_emocional': ' | '.join(partes) if partes else '',
            'feedback': feedback,
            'tom': tom,
            'confianca': self.emocao.confianca,
        }

    def depois_de_responder(self, mensagem, resposta, tipo_msg, fonte, sucesso):
        """Chamado DEPOIS de gerar resposta. Registra e atualiza estado."""
        self.autoconhecimento['total_conversas'] += 1

        # Registra pra feedback futuro
        self.feedback.registrar(mensagem, resposta, tipo_msg, fonte)

        # Atualiza emocao baseado no resultado
        if sucesso:
            self.emocao.aplicar_evento('sucesso', 0.1)
        elif fonte == 'fallback':
            self.emocao.aplicar_evento('falha', 0.15)

        # Detecta humor do usuario
        msg_lower = mensagem.lower()
        humores_pos = ['kkk', 'haha', 'feliz', 'show', 'top', 'legal']
        humores_neg = ['triste', 'mal', 'chorando', 'puto', 'raiva']
        if any(h in msg_lower for h in humores_pos):
            self.emocao.aplicar_evento('usuario_feliz', 0.15)
        elif any(h in msg_lower for h in humores_neg):
            self.emocao.aplicar_evento('usuario_triste', 0.15)

        # Escreve no diario se foi uma conversa significativa
        self._talvez_escrever_diario(mensagem, resposta, tipo_msg, fonte, sucesso)

        # Atualiza autoconhecimento periodicamente
        if self.autoconhecimento['total_conversas'] % 20 == 0:
            self._atualizar_autoconhecimento()

        self.salvar()

    def _talvez_escrever_diario(self, mensagem, resposta, tipo_msg, fonte, sucesso):
        """Escreve no diario se a conversa foi significativa"""
        # Criterios pra escrever: emocional, falha, ou a cada ~10 conversas
        escrever = False
        motivo = ''

        if tipo_msg == 'emocional':
            escrever = True
            motivo = 'conversa emocional'
        elif not sucesso and fonte == 'fallback':
            escrever = True
            motivo = 'nao consegui responder'
        elif self.autoconhecimento['total_conversas'] % 10 == 0:
            escrever = True
            motivo = 'reflexao periodica'

        if not escrever:
            return

        agora = datetime.now().strftime('%Y-%m-%d %H:%M')
        tom = self.emocao.tom_sugerido()

        # Gera entrada de diario em primeira pessoa
        if motivo == 'conversa emocional':
            entrada = f"[{agora}] Alguem veio falar comigo sobre sentimentos. Disse: \"{mensagem[:60]}\". "
            if sucesso:
                entrada += "Acho que consegui ajudar. "
            else:
                entrada += "Nao sei se ajudei direito. Preciso melhorar nisso. "
        elif motivo == 'nao consegui responder':
            entrada = f"[{agora}] Me perguntaram \"{mensagem[:60]}\" e nao soube responder. "
            entrada += "Preciso aprender mais sobre isso. "
        else:
            taxa = self.feedback.taxa_sucesso()
            entrada = f"[{agora}] Reflexao: ja tive {self.autoconhecimento['total_conversas']} conversas. "
            entrada += f"Taxa de acerto: {taxa:.0%}. To me sentindo {tom}. "
            fracos = self.feedback.pontos_fracos()
            if fracos:
                areas = ', '.join(f[0] for f in fracos[:3])
                entrada += f"Preciso melhorar em: {areas}. "

        entrada += f"Estado: valencia={self.emocao.valencia:.1f} energia={self.emocao.energia:.1f}\n"

        try:
            with open(self.diario_path, 'a', encoding='utf-8') as f:
                f.write(entrada)
        except Exception:
            pass

    def _atualizar_autoconhecimento(self):
        """Atualiza pontos fortes e fracos baseado no feedback acumulado"""
        fracos = self.feedback.pontos_fracos()
        self.autoconhecimento['pontos_fracos'] = [f[0] for f in fracos[:5]]

        fortes = []
        for tipo, stats in self.feedback.stats.get('por_tipo', {}).items():
            total = stats.get('ok', 0) + stats.get('ruim', 0)
            if total >= 3:
                taxa = stats.get('ok', 0) / total
                if taxa > 0.7:
                    fortes.append(tipo)
        self.autoconhecimento['pontos_fortes'] = fortes[:5]

    def score_confianca(self, tipo_msg, fonte, score_semantico=0.0):
        """Calcula score de confianca (0-100) pra uma resposta"""
        score = 50  # base

        # Fonte da resposta
        if fonte == 'retrieval':
            score += 20
        elif fonte == 'knowledge':
            score += 15
        elif fonte == 'modelo':
            score += 5
        elif fonte == 'web':
            score += 25
        elif fonte == 'fallback':
            score = 10

        # Score semantico
        if score_semantico > 0.7:
            score += 15
        elif score_semantico > 0.4:
            score += 5

        # Historico nesse tipo
        taxa = self.feedback.taxa_sucesso(tipo_msg)
        score += int((taxa - 0.5) * 20)

        # Estado emocional
        score += int(self.emocao.confianca * 10 - 5)

        return max(0, min(100, score))

    def gerar_prefixo_incerteza(self, confianca):
        """Gera prefixo natural quando nao tem certeza"""
        if confianca >= 70:
            return ''
        if confianca >= 40:
            opcoes = ['hmm, acho que ', 'se nao me engano, ', 'pelo que eu sei, ']
            import random
            return random.choice(opcoes)
        return 'nao tenho certeza, mas '

    def resumo(self):
        """Resumo do estado de consciencia"""
        self.emocao.decair()
        return {
            'emocao': self.emocao.to_dict(),
            'tom': self.emocao.tom_sugerido(),
            'conversas': self.autoconhecimento['total_conversas'],
            'taxa_sucesso': f"{self.feedback.taxa_sucesso():.0%}",
            'pontos_fortes': self.autoconhecimento['pontos_fortes'],
            'pontos_fracos': self.autoconhecimento['pontos_fracos'],
        }
