"""
Auto-Reflexão da Keilinks (Fase 4)
Antes de responder, avalia:
1. Eu sei sobre isso? (knowledge + retrieval)
2. Preciso pensar mais? (análise da pergunta)
3. A resposta faz sentido? (validação)
4. Devo ajustar meu tom? (humor do usuário)

Cria uma "cadeia de pensamento" interna.
"""

import re


class Reflexao:
    """Processa a pergunta antes de decidir como responder"""

    def analisar(self, mensagem: str, contexto_memoria: str = '') -> dict:
        """
        Analisa a mensagem e retorna metadados para guiar a resposta.
        Retorna dict com:
          - tipo: 'saudacao', 'pergunta_pessoal', 'pergunta_factual',
                  'pedido_ajuda', 'emocional', 'conversa_casual', 'comando'
          - precisa_web: bool
          - urgencia: 'baixa', 'media', 'alta'
          - humor_detectado: str
          - topico_extraido: str
        """
        msg = mensagem.lower().strip()

        resultado = {
            'tipo': 'conversa_casual',
            'precisa_web': False,
            'urgencia': 'baixa',
            'humor_detectado': 'neutro',
            'topico_extraido': '',
        }

        # ─── Tipo da mensagem ──────────────────────────────────────────

        # Saudação
        saudacoes = ['oi', 'ola', 'hey', 'bom dia', 'boa tarde', 'boa noite',
                     'e ai', 'eai', 'fala', 'salve']
        if msg in saudacoes or any(msg.startswith(s + ' ') for s in saudacoes):
            resultado['tipo'] = 'saudacao'
            return resultado

        # Despedida
        despedidas = ['tchau', 'ate mais', 'ate logo', 'flw', 'vlw', 'falou']
        if msg in despedidas or any(msg.startswith(d) for d in despedidas):
            resultado['tipo'] = 'despedida'
            return resultado

        # Emocional
        emocoes = ['triste', 'feliz', 'raiva', 'ansioso', 'medo', 'chorando',
                   'puto', 'irritado', 'cansado', 'to mal', 'to bem']
        if any(e in msg for e in emocoes):
            resultado['tipo'] = 'emocional'
            resultado['urgencia'] = 'alta'
            resultado['humor_detectado'] = self._detectar_humor(msg)
            return resultado

        # Pergunta pessoal (sobre o Vitor, família, Keilinks)
        pessoal = ['vitor', 'keila', 'adriano', 'juliene', 'natalia',
                   'namorada', 'pai', 'mae', 'irma', 'familia',
                   'keilinks', 'seu criador', 'te criou', 'seu nome',
                   'voce é', 'você é', 'quem é voce', 'quem é você']
        if any(p in msg for p in pessoal):
            resultado['tipo'] = 'pergunta_pessoal'
            resultado['urgencia'] = 'media'
            return resultado

        # Pergunta factual (precisa de web)
        factual = ['o que é', 'quem é', 'quem foi', 'quando foi', 'quando é',
                   'onde fica', 'como funciona', 'qual é', 'quantos',
                   'quanto custa', 'preço', 'capital de', 'população']
        if any(f in msg for f in factual):
            resultado['tipo'] = 'pergunta_factual'
            resultado['precisa_web'] = True
            resultado['urgencia'] = 'media'
            resultado['topico_extraido'] = self._extrair_topico(msg)
            return resultado

        # Pedido de ajuda (programação, técnico)
        ajuda = ['me ajuda', 'como faz', 'como faço', 'me ensina',
                 'bug', 'erro', 'código', 'programar', 'função']
        if any(a in msg for a in ajuda):
            resultado['tipo'] = 'pedido_ajuda'
            resultado['urgencia'] = 'media'
            return resultado

        # Comando direto
        comandos = ['pesquisa', 'busca', 'procura', 'acha pra mim']
        if any(c in msg for c in comandos):
            resultado['tipo'] = 'comando'
            resultado['precisa_web'] = True
            resultado['topico_extraido'] = self._extrair_topico(msg)
            return resultado

        return resultado

    def _detectar_humor(self, msg: str) -> str:
        positivos = ['feliz', 'to bem', 'animado', 'boa', 'legal', 'show']
        negativos = ['triste', 'to mal', 'chorando', 'sozinho']
        raiva = ['puto', 'irritado', 'raiva', 'odeio']

        if any(p in msg for p in positivos):
            return 'feliz'
        if any(n in msg for n in negativos):
            return 'triste'
        if any(r in msg for r in raiva):
            return 'raiva'
        return 'neutro'

    def _extrair_topico(self, msg: str) -> str:
        """Extrai o tópico principal da pergunta"""
        patterns = [
            r'(?:o que (?:é|sao|significa))\s+(?:um |uma |o |a )?(.+)',
            r'(?:quem (?:é|foi|sao))\s+(?:o |a )?(.+)',
            r'(?:(?:me |)(?:fala|conta|explica) (?:sobre|do|da|de))\s+(.+)',
            r'(?:como funciona)\s+(?:o |a |um |uma )?(.+)',
            r'(?:pesquisa|busca|procura)\s+(.+)',
        ]
        for pat in patterns:
            match = re.search(pat, msg)
            if match:
                return match.group(1).strip().rstrip('?.,!')
        return msg.strip()

    def validar_resposta(self, resposta: str) -> bool:
        """Valida se a resposta é aceitável"""
        if not resposta or len(resposta) < 2:
            return False
        # Muita repetição
        if len(set(resposta.split())) < len(resposta.split()) * 0.2:
            return False
        return True
