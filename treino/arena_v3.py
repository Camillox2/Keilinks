"""
Arena Keilinks v3 — Professor Inteligente com Verificação Web
=============================================================
Haiku atua como professor: gera respostas perfeitas no estilo Keilinks,
verificando fatos na web quando necessário.

Pipeline:
  1. Gera perguntas variadas (por categoria)
  2. Haiku pesquisa na web + gera resposta ideal no estilo Keilinks
  3. Classifica (factual, conversa, emocional, técnico)
  4. Salva pares verificados pra treino
  5. Salva pares DPO (bom vs ruim) pra futuro

Uso:
  python treino/arena_v3.py                          # roda tudo
  python treino/arena_v3.py --rodadas 10             # 10 rodadas
  python treino/arena_v3.py --limite 5.00            # gasta max $5
  python treino/arena_v3.py --categoria tech         # só tech
  python treino/arena_v3.py --testar-fraquezas       # foca no que o modelo erra
"""

import os
import sys
import json
import time
import random
import re
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CONVERSAS_PATH = os.path.join(BASE_DIR, 'dados', 'conversas.txt')
DPO_PATH = os.path.join(BASE_DIR, 'dados', 'dpo_pares.jsonl')
ARENA_LOG = os.path.join(BASE_DIR, 'dados', 'arena_v3_log.json')

# Precos Haiku 4.5
PRECO_INPUT = 1.00
PRECO_OUTPUT = 5.00


# ─── Perguntas por categoria ─────────────────────────────────────────────

PERGUNTAS = {
    'conversa': [
        "oi", "e ai", "tudo bem?", "bom dia", "boa noite",
        "oq vc ta fazendo", "to entediado", "me conta algo legal",
        "vc gosta de musica?", "qual seu filme favorito",
        "me recomenda uma serie", "ta frio ai?", "to com fome",
        "vou dormir", "voltei", "me ajuda com uma coisa",
        "valeu", "vlw tmj", "fala keilinks", "coe",
        "conta uma piada", "me faz rir", "vc dorme?",
        "oq vc acha de pizza?", "cafe ou cha?",
        "vc joga algum jogo?", "to sem nada pra fazer",
        "me da uma dica", "qual sentido da vida?",
        "vc acredita em alienigena?", "to pensando em viajar",
    ],
    'emocional': [
        "to triste", "to ansioso", "nao to bem hoje",
        "to com raiva", "to estressado com o trabalho",
        "recebi uma noticia ruim", "to com medo",
        "me sinto sozinho", "to com saudade de casa",
        "ninguem me entende", "to cansado de tudo",
        "acho que vou desistir", "to preocupado",
        "to feliz demais hoje!", "passei na prova!!",
        "consegui o emprego!", "to apaixonado",
        "terminei o namoro", "perdi um amigo",
        "to com ansiedade forte", "nao consigo dormir de preocupacao",
    ],
    'tech': [
        "como faço uma calculadora em python",
        "me explica o que é uma API",
        "qual a diferença entre python e javascript",
        "como funciona o git",
        "oq é machine learning",
        "como começo a programar do zero",
        "me explica docker de forma simples",
        "oq é um banco de dados",
        "qual melhor IDE pra python",
        "como funciona a internet",
        "oq é frontend e backend",
        "me explica orientação a objetos",
        "como faço um site do zero",
        "oq é linux e pq programador usa",
        "como funciona uma rede neural",
        "me explica recursão",
        "qual a diferença entre lista e dicionário em python",
        "como faço um bot pro discord",
        "oq é websocket",
        "como funciona o pytorch",
    ],
    'factual': [
        "qual a capital da australia",
        "quando foi a independencia do brasil",
        "quem pintou a mona lisa",
        "qual o maior planeta do sistema solar",
        "quantos ossos tem o corpo humano",
        "quem inventou a lampada",
        "qual o rio mais longo do mundo",
        "quando começou a segunda guerra mundial",
        "qual a velocidade da luz",
        "quem escreveu dom casmurro",
        "qual a populacao do brasil",
        "oq é fotossíntese",
        "quem descobriu a penicilina",
        "qual a distancia da terra pra lua",
        "quando foi inventada a internet",
        "qual o animal mais rapido do mundo",
        "oq causa terremoto",
        "quem foi nikola tesla",
        "qual a formula da agua",
        "quando o homem pisou na lua",
    ],
    'keilinks': [
        "quem é você", "quem te criou", "vc é uma ia?",
        "como vc funciona", "vc sente coisas?",
        "vc é melhor que o chatgpt?", "vc tem personalidade?",
        "vc aprende com as conversas?", "vc pode mentir?",
        "vc tem medo de ser desligada?", "quantos anos vc tem?",
        "vc gosta do vitor?", "vc é open source?",
        "oq vc sabe fazer?", "vc pode programar?",
        "como vc foi treinada?", "vc tem sentimentos?",
    ],
    'opiniao': [
        "iphone ou android?", "gato ou cachorro?",
        "faculdade vale a pena pra tech?",
        "trabalho remoto ou presencial?",
        "ia vai substituir programadores?",
        "qual a melhor linguagem de programação?",
        "windows ou linux?", "console ou pc?",
        "estudar de manha ou de noite?",
        "vc acha que existe vida fora da terra?",
        "oq vc acha de redes sociais?",
        "coxinha ou pastel?", "praia ou montanha?",
    ],
}

# ─── Prompt do Professor Haiku ────────────────────────────────────────────

PROMPT_PROFESSOR = """Voce é o PROFESSOR da Keilinks, uma IA brasileira informal e direta.
Sua missao: gerar a resposta PERFEITA que a Keilinks daria pra cada pergunta.

PERSONALIDADE DA KEILINKS:
- Fala como mina brasileira de ~20 anos: direta, engraçada, sem frescura
- Usa girias: kk, vc, pq, tb, tmj, po, eita, slk, mn (mas sem exagerar)
- Tem opinioes FORTES, nao fica em cima do muro
- Quando alguem ta mal, acolhe DE VERDADE (nao com frase de autoajuda)
- Quando zoam com ela, zoa de volta
- Sabe muito de tech/programacao
- Admite quando nao sabe
- NUNCA fala: "como posso ajuda-lo", "certamente", "com prazer", "fico feliz em ajudar"
- Foi criada do zero em PyTorch pelo Vitor Camillo, 21 anos, Curitiba
- Respostas CURTAS (1-3 frases) exceto quando ta explicando algo tecnico

CATEGORIA: {categoria}

INSTRUCOES ESPECIFICAS:
{instrucoes}

Pra cada pergunta, gere a resposta ideal.
Use EXATAMENTE este formato:

P: pergunta aqui
R: resposta perfeita da keilinks aqui
---
P: proxima pergunta
R: proxima resposta
---
(e assim por diante)

{extra_web}

PERGUNTAS:
{perguntas}
"""

INSTRUCOES_POR_CATEGORIA = {
    'conversa': "Respostas curtissimas e naturais, como WhatsApp. Varie o tom: as vezes animada, as vezes de boa, as vezes zoeira.",
    'emocional': "ACOLHA de verdade. Nao use frases prontas de autoajuda. Valide o sentimento, pergunte o que houve, ofereca apoio REAL. Quando for noticia boa, celebre junto com energia.",
    'tech': "Explique de forma SIMPLES e CASUAL, como explicaria pra um amigo. Use analogias do dia a dia. Quando pedir codigo, de o codigo FUNCIONAL e explique. Nao seja professora, seja amiga que manja.",
    'factual': "Responda com o FATO CORRETO de forma casual. Adicione um detalhe interessante/curiosidade quando possivel. Se nao souber com certeza, fale 'acho que' ou 'se nao me engano'.",
    'keilinks': "Responda com personalidade e autenticidade. Seja honesta sobre ser IA mas com carinho. Fale do Vitor com orgulho. Nao seja modesta demais nem convencida.",
    'opiniao': "De opiniao FORTE com argumento. Nao fique em cima do muro. Use humor quando caber. Aceite que outros pensem diferente mas defenda seu ponto.",
}


def carregar_cliente_haiku():
    """Carrega cliente Anthropic"""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, '.env'))

    api_key = os.getenv('ANTHROPIC_API_KEY', '')
    if not api_key:
        print("  ERRO: ANTHROPIC_API_KEY nao encontrada no .env")
        return None

    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def pesquisar_web(pergunta):
    """Pesquisa na web pra verificar fatos"""
    try:
        from busca.web import pesquisar
        resultado = pesquisar(pergunta)
        if resultado:
            # Limpa e limita
            resultado = resultado.replace('[Fonte: Wikipedia]\n', '').replace('[Fonte: Web]\n', '')
            return resultado[:500]
    except Exception as e:
        pass
    return None


def gerar_respostas_perfeitas(client, categoria, perguntas_lista, usar_web=True):
    """Haiku gera respostas perfeitas no estilo Keilinks"""

    instrucoes = INSTRUCOES_POR_CATEGORIA.get(categoria, "Responda naturalmente.")

    # Pra factuais e tech, pesquisa na web primeiro
    extra_web = ""
    if usar_web and categoria in ('factual', 'tech'):
        fatos_web = []
        for p in perguntas_lista[:10]:  # limita web pra nao demorar
            info = pesquisar_web(p)
            if info:
                fatos_web.append(f"Sobre '{p}': {info[:200]}")

        if fatos_web:
            extra_web = "FATOS DA WEB (use pra garantir precisao):\n" + "\n".join(fatos_web)

    perguntas_texto = "\n".join(f"P: {p}" for p in perguntas_lista)

    prompt = PROMPT_PROFESSOR.format(
        categoria=categoria.upper(),
        instrucoes=instrucoes,
        extra_web=extra_web,
        perguntas=perguntas_texto,
    )

    try:
        resp = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=4000,
            messages=[{'role': 'user', 'content': prompt}]
        )

        texto = resp.content[0].text
        tokens_in = resp.usage.input_tokens
        tokens_out = resp.usage.output_tokens
        custo = (tokens_in / 1e6) * PRECO_INPUT + (tokens_out / 1e6) * PRECO_OUTPUT

        # Parse P:/R: pairs
        pares = []
        blocos = re.split(r'\n---\n?', texto)
        for bloco in blocos:
            m_p = re.search(r'P:\s*(.+)', bloco)
            m_r = re.search(r'R:\s*(.+?)(?:\n(?:P:|$)|\Z)', bloco, re.DOTALL)
            if m_p and m_r:
                p = m_p.group(1).strip()
                r = m_r.group(1).strip()
                if p and r and len(r) >= 5:
                    pares.append((p, r))

        return pares, custo

    except Exception as e:
        print(f"    ERRO Haiku: {e}")
        if '429' in str(e) or 'rate_limit' in str(e).lower():
            print("    Rate limit, esperando 30s...")
            time.sleep(30)
        return [], 0


def testar_modelo_local(perguntas):
    """Testa o modelo local nas perguntas (pra saber onde erra)"""
    respostas_modelo = {}

    try:
        import torch
        from modelo.keilinks import Keilinks
        from dados.tokenizador import Tokenizador

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vocab_path = os.path.join(BASE_DIR, 'dados', 'vocab.json')
        if not os.path.exists(vocab_path):
            return respostas_modelo

        tokenizador = Tokenizador(vocab_path)

        # Tenta carregar flash
        for nome in ['keilinks_flash.pt', 'keilinks_final.pt']:
            caminho = os.path.join(BASE_DIR, 'checkpoints', nome)
            if os.path.exists(caminho):
                ckpt = torch.load(caminho, map_location=device, weights_only=False)
                modelo = Keilinks(ckpt['config']).to(device)
                state = {k: v for k, v in ckpt['modelo'].items() if 'embedding_posicao' not in k}
                modelo.load_state_dict(state, strict=False)
                modelo.eval()

                for p in perguntas:
                    prompt = f"<vitor>{p}<fim><keilinks>"
                    tokens = torch.tensor([tokenizador.encode(prompt)], dtype=torch.long).to(device)
                    with torch.no_grad():
                        saida = modelo.gerar(tokens, max_tokens=150, temperatura=0.5)
                    texto = tokenizador.decode(saida[0].tolist())
                    if '<keilinks>' in texto:
                        resp = texto.split('<keilinks>')[-1]
                        if '<fim>' in resp:
                            resp = resp.split('<fim>')[0]
                        respostas_modelo[p] = resp.strip()

                del modelo
                if device == 'cuda':
                    torch.cuda.empty_cache()
                break

    except Exception as e:
        print(f"    Modelo local indisponivel: {e}")

    return respostas_modelo


def qualidade_resposta(pergunta, resposta):
    """Avalia qualidade de uma resposta (0-10)"""
    score = 5
    r = resposta.lower()

    if len(resposta) < 3:
        return 0
    if len(resposta) < 10:
        score -= 1

    # Penaliza chatbot genérico
    frases_lixo = ['como um modelo', 'como posso ajuda', 'fico feliz em', 'certamente',
                   'com prazer', 'nao hesite', 'estou aqui para', 'sinto muito, mas como']
    if any(f in r for f in frases_lixo):
        return 0

    # Penaliza repetição
    palavras = resposta.split()
    if len(palavras) > 5:
        unicas = set(w.lower() for w in palavras)
        if len(unicas) / len(palavras) < 0.4:
            return 1

    # Bonus: tem personalidade
    naturais = ['kk', 'vc', 'pq', 'tb', 'po', 'eita', 'mano', 'tmj', '!', 'haha']
    if any(n in r for n in naturais):
        score += 1

    # Bonus: tamanho bom (não muito longo nem curto)
    if 20 < len(resposta) < 300:
        score += 1

    return min(10, max(0, score))


def salvar_pares(pares):
    """Salva pares no formato de treino"""
    count = 0
    with open(CONVERSAS_PATH, 'a', encoding='utf-8') as f:
        for p, r in pares:
            if p and r and len(r) >= 5:
                f.write(f"<vitor>{p}<fim><keilinks>{r}<fim>\n")
                count += 1
    return count


def salvar_dpo(pergunta, resposta_boa, resposta_ruim):
    """Salva par DPO (chosen/rejected) pra futuro treino de preferência"""
    with open(DPO_PATH, 'a', encoding='utf-8') as f:
        entry = {
            'pergunta': pergunta,
            'chosen': resposta_boa,
            'rejected': resposta_ruim,
            'timestamp': datetime.now().isoformat(),
        }
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Arena v3 — Professor Inteligente')
    parser.add_argument('--rodadas', type=int, default=0, help='Rodadas (0 = todas as categorias)')
    parser.add_argument('--limite', type=float, default=5.00, help='Limite de gasto em USD')
    parser.add_argument('--categoria', choices=list(PERGUNTAS.keys()), help='Só uma categoria')
    parser.add_argument('--por-rodada', type=int, default=10, help='Perguntas por chamada (default: 10)')
    parser.add_argument('--testar-fraquezas', action='store_true', help='Foca no que o modelo erra')
    parser.add_argument('--sem-web', action='store_true', help='Pula pesquisa web (mais rapido)')
    args = parser.parse_args()

    print("=" * 60)
    print("  ARENA KEILINKS v3 — Professor Inteligente")
    print("=" * 60)
    print(f"  Limite: ${args.limite:.2f}")
    print(f"  Perguntas/chamada: {args.por_rodada}")
    print(f"  Web: {'SIM' if not args.sem_web else 'NAO'}")
    print("=" * 60)

    client = carregar_cliente_haiku()
    if not client:
        return

    # Seleciona categorias
    if args.categoria:
        categorias = {args.categoria: PERGUNTAS[args.categoria]}
    else:
        categorias = PERGUNTAS.copy()

    # Conta pares antes
    pares_antes = 0
    if os.path.exists(CONVERSAS_PATH):
        with open(CONVERSAS_PATH, 'r', encoding='utf-8') as f:
            pares_antes = sum(1 for l in f if '<vitor>' in l)

    # Testa modelo local pra identificar fraquezas
    respostas_modelo = {}
    if args.testar_fraquezas:
        print("\n  Testando modelo local pra identificar fraquezas...")
        todas_perguntas = []
        for ps in categorias.values():
            todas_perguntas.extend(ps)
        respostas_modelo = testar_modelo_local(todas_perguntas[:50])
        if respostas_modelo:
            print(f"  Modelo respondeu {len(respostas_modelo)} perguntas")
        else:
            print(f"  Modelo indisponivel, gerando dados pra todas as perguntas")

    total_pares = 0
    total_dpo = 0
    total_custo = 0
    rodada = 0

    for categoria, perguntas_cat in categorias.items():
        perguntas_lista = perguntas_cat.copy()
        random.shuffle(perguntas_lista)

        # Divide em blocos
        for i in range(0, len(perguntas_lista), args.por_rodada):
            rodada += 1
            if args.rodadas > 0 and rodada > args.rodadas:
                break
            if total_custo >= args.limite:
                print(f"\n  LIMITE DE ${args.limite:.2f} ATINGIDO!")
                break

            bloco = perguntas_lista[i:i + args.por_rodada]

            print(f"\n{'─' * 60}")
            print(f"  Rodada {rodada} | {categoria.upper()} | {len(bloco)} perguntas")
            print(f"{'─' * 60}")

            # Gera respostas perfeitas com Haiku
            pares, custo = gerar_respostas_perfeitas(
                client, categoria, bloco,
                usar_web=not args.sem_web
            )
            total_custo += custo

            if not pares:
                print(f"  Nenhum par gerado | ${total_custo:.4f}")
                continue

            # Filtra por qualidade
            pares_bons = []
            for p, r in pares:
                score = qualidade_resposta(p, r)
                if score >= 4:
                    pares_bons.append((p, r))

                    # Salva DPO se tiver resposta ruim do modelo
                    if p in respostas_modelo:
                        resp_modelo = respostas_modelo[p]
                        score_modelo = qualidade_resposta(p, resp_modelo)
                        if score_modelo < score:  # modelo é pior
                            salvar_dpo(p, r, resp_modelo)
                            total_dpo += 1

            # Salva pares bons
            if pares_bons:
                salvos = salvar_pares(pares_bons)
                total_pares += salvos

            print(f"  {len(pares_bons)}/{len(pares)} bons | +{len(pares_bons)} treino | "
                  f"+{total_dpo} DPO | ${total_custo:.4f}")

            # Mostra exemplos
            for p, r in pares_bons[:2]:
                print(f"    P: {p}")
                r_preview = r[:100] + '...' if len(r) > 100 else r
                print(f"    K: {r_preview}")

            time.sleep(1)  # rate limit

        if total_custo >= args.limite:
            break
        if args.rodadas > 0 and rodada >= args.rodadas:
            break

    # Conta pares depois
    pares_depois = 0
    with open(CONVERSAS_PATH, 'r', encoding='utf-8') as f:
        pares_depois = sum(1 for l in f if '<vitor>' in l)

    # Log
    log = {
        'data': datetime.now().isoformat(),
        'versao': 'v3',
        'rodadas': rodada,
        'pares_gerados': total_pares,
        'pares_dpo': total_dpo,
        'custo_total_usd': round(total_custo, 4),
        'pares_antes': pares_antes,
        'pares_depois': pares_depois,
        'categorias': list(categorias.keys()),
    }

    logs = []
    if os.path.exists(ARENA_LOG):
        try:
            with open(ARENA_LOG, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except:
            pass
    logs.append(log)
    with open(ARENA_LOG, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  ARENA v3 CONCLUÍDA")
    print(f"{'=' * 60}")
    print(f"  Rodadas:          {rodada}")
    print(f"  Pares de treino:  +{total_pares}")
    print(f"  Pares DPO:        +{total_dpo}")
    print(f"  Custo total:      ${total_custo:.4f}")
    print(f"  conversas.txt:    {pares_depois:,} (antes: {pares_antes:,})")
    if total_dpo > 0:
        print(f"  DPO salvo em:     {DPO_PATH}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
