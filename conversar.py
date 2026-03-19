import torch
import sys
import os
import threading
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modelo.keilinks import Keilinks
from dados.tokenizador import Tokenizador
from cerebro.consciencia import Consciencia
from cerebro.memoria import MemoriaLongoPrazo
from busca.web import BuscadorWeb

# Variáveis globais para controlar o tempo e o acesso à placa de vídeo
ultima_interacao = time.time()
gpu_lock = threading.Lock()

def motor_subconsciente(modelo, tokenizador, consciencia, device):
    global ultima_interacao
    while True:
        time.sleep(30)
        agora = time.time()
        
        # Ativa o subconsciente se não falar com ela há mais de 5 minutos
        if agora - ultima_interacao > 300:
            with gpu_lock:
                consciencia.emocao.decair()
                consciencia.emocao.aplicar_evento('inatividade', 0.1)
                
                prompt_pensamento = "<sistema>Reflita internamente sobre suas conversas recentes, quem você é, e como está se sentindo agora.<fim><keilinks>"
                tokens = torch.tensor([tokenizador.encode(prompt_pensamento)], dtype=torch.long).to(device)
                
                saida = modelo.gerar(tokens, max_tokens=100, temperatura=0.9, top_p=0.95)
                texto_gerado = tokenizador.decode(saida[0].tolist())
                
                resposta_bruta = texto_gerado[len(prompt_pensamento):]
                if '<fim>' in resposta_bruta:
                    reflexao = resposta_bruta.split('<fim>')[0]
                else:
                    reflexao = resposta_bruta
                
                consciencia._talvez_escrever_diario("Pensamento interno", reflexao.strip(), 'emocional', 'subconsciente', True)
                consciencia.salvar()
            
            ultima_interacao = time.time()

def conversar():
    global ultima_interacao
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for path in ['checkpoints/keilinks_flash.pt', 'checkpoints/keilinks_final.pt', 'checkpoints/keilinks_ultra.pt']:
        if os.path.exists(path):
            checkpoint_path = path
            break
    else:
        print("Keilinks ainda não foi treinada. Execute: python treino/treinar.py")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizador = Tokenizador('dados/vocab.json')
    modelo = Keilinks(checkpoint['config']).to(device)
    
    state = {k: v for k, v in checkpoint['modelo'].items() if 'embedding_posicao' not in k}
    modelo.load_state_dict(state, strict=False)
    modelo.eval()

    # Inicializa todos os módulos do cérebro
    consciencia = Consciencia('dados/')
    memoria_longo_prazo = MemoriaLongoPrazo()
    buscador_web = BuscadorWeb()
    
    historico_conversa = ""
    ctx_max = checkpoint['config'].get('contexto_max', 2048)

    # Inicia a thread do subconsciente
    thread_pensamento = threading.Thread(target=motor_subconsciente, args=(modelo, tokenizador, consciencia, device), daemon=True)
    thread_pensamento.start()

    print("Keilinks online (Motor Autônomo, RAG e Internet Ativados). Digite 'sair' para encerrar.\n")
    print("-" * 50)

    while True:
        entrada = input("Vitor: ").strip()
        if entrada.lower() in ['sair', 'exit', 'quit']:
            print("Keilinks: Até mais.")
            break
        if not entrada:
            continue

        ultima_interacao = time.time()

        estado_ctx = consciencia.antes_de_responder(entrada, 'chat')
        emocao_txt = estado_ctx['contexto_emocional']

        # 1. Busca memórias antigas relevantes (RAG)
        lembrancas = memoria_longo_prazo.buscar_memoria(entrada, top_k=2)
        contexto_memoria = ""
        if lembrancas:
            contexto_memoria = " Memórias do passado com o Vitor: " + " | ".join(lembrancas) + "."

        # 2. Pesquisa na Web em tempo real
        gatilhos_web = ['hoje', 'agora', 'notícia', 'noticias', 'quem ganhou', 'clima', 'tempo em', 'preço do', 'cotação', 'atual']
        precisa_web = any(gatilho in entrada.lower() for gatilho in gatilhos_web)
        
        contexto_web = ""
        if precisa_web:
            print("  [A Keilinks está pesquisando na web...]")
            resultados_pesquisa = buscador_web.pesquisar(entrada)
            if resultados_pesquisa:
                contexto_web = f" Informação da internet em tempo real para ajudar na resposta: {resultados_pesquisa}."

        # Injeta as emoções, lembranças e dados da internet no prompt do sistema de forma invisível
        system_prompt = f"<sistema>Você é Keilinks, a IA pessoal e amigável do Vitor. {emocao_txt}{contexto_memoria}{contexto_web}<fim>"
        historico_conversa += f"<vitor>{entrada}<fim><keilinks>"
        prompt_completo = system_prompt + historico_conversa

        with gpu_lock:
            tokens = torch.tensor([tokenizador.encode(prompt_completo)], dtype=torch.long).to(device)

            if tokens.shape[1] > ctx_max - 150:
                tokens = tokens[:, -(ctx_max - 150):]
                historico_conversa = tokenizador.decode(tokens[0].tolist())
                if '<sistema>' in historico_conversa:
                    historico_conversa = historico_conversa.split('<sistema>')[-1]
                if '<fim>' in historico_conversa:
                    historico_conversa = historico_conversa.split('<fim>', 1)[-1]

            saida = modelo.gerar(tokens, max_tokens=150, temperatura=0.85, top_p=0.92)
            texto_gerado = tokenizador.decode(saida[0].tolist())

            resposta_bruta = texto_gerado[len(prompt_completo):]
            
            if '<fim>' in resposta_bruta:
                resposta = resposta_bruta.split('<fim>')[0]
            else:
                resposta = resposta_bruta

        print(f"Keilinks: {resposta.strip()}\n")

        # 3. Guarda a nova memória para o futuro (se for uma frase de bom tamanho)
        if len(entrada) > 15:
            threading.Thread(target=memoria_longo_prazo.adicionar_memoria, args=(entrada,)).start()

        historico_conversa += f"{resposta.strip()}<fim>"
        consciencia.depois_de_responder(entrada, resposta.strip(), 'chat', 'modelo', True)

if __name__ == '__main__':
    conversar()