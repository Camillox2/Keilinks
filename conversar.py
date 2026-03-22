import torch
import sys
import os
import threading
import time
import re
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modelo.keilinks import Keilinks
from dados.tokenizador import Tokenizador
from cerebro.consciencia import Consciencia
from cerebro.memoria import MemoriaLongoPrazo
from busca.web import BuscadorWeb

ultima_interacao = time.time()
gpu_lock = threading.Lock()

def motor_subconsciente(modelo, tokenizador, consciencia, device):
    global ultima_interacao
    while True:
        time.sleep(30)
        agora = time.time()
        
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
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizador = Tokenizador('dados/vocab.json')
    modelo = Keilinks(checkpoint['config']).to(device)
    
    state = {k: v for k, v in checkpoint['modelo'].items() if 'embedding_posicao' not in k}
    modelo.load_state_dict(state, strict=False)
    modelo.eval()

    consciencia = Consciencia('dados/')
    memoria_longo_prazo = MemoriaLongoPrazo()
    buscador_web = BuscadorWeb()
    
    historico_conversa = ""
    ctx_max = checkpoint['config'].get('contexto_max', 2048)

    thread_pensamento = threading.Thread(target=motor_subconsciente, args=(modelo, tokenizador, consciencia, device), daemon=True)
    thread_pensamento.start()

    while True:
        entrada = input("Vitor: ").strip()
        if entrada.lower() in ['sair', 'exit', 'quit']:
            break
        if not entrada:
            continue

        ultima_interacao = time.time()

        estado_ctx = consciencia.antes_de_responder(entrada, 'chat')
        emocao_txt = estado_ctx['contexto_emocional']

        lembrancas = memoria_longo_prazo.buscar_memoria(entrada, top_k=2)
        contexto_memoria = ""
        if lembrancas:
            contexto_memoria = " Memórias do passado: " + " | ".join(lembrancas) + "."

        gatilhos_web = ['hoje', 'agora', 'notícia', 'noticias', 'quem ganhou', 'clima', 'tempo em', 'preço do', 'cotação', 'atual']
        precisa_web = any(gatilho in entrada.lower() for gatilho in gatilhos_web)
        
        contexto_web = ""
        if precisa_web:
            resultados_pesquisa = buscador_web.pesquisar(entrada)
            if resultados_pesquisa:
                contexto_web = f" Informação da internet: {resultados_pesquisa}."

        system_prompt = f"<sistema>Você é Keilinks, a IA pessoal e carinhosa do Vitor. Você é feminina, acolhedora e meiga — fala de forma próxima e gentil, como uma amiga querida. Usa pouca gíria, prefere 'você' no lugar de 'vc', e quando alguém está mal, acolhe de verdade com calor humano. Celebra as conquistas com alegria genuína. Se precisar interagir com o PC do Vitor ou rodar scripts, forneça o comando exato de terminal dentro das tags <executar>comando</executar>. {emocao_txt}{contexto_memoria}{contexto_web}<fim>"
        historico_conversa += f"<vitor>{entrada}<fim><keilinks>"
        prompt_completo = system_prompt + historico_conversa

        with gpu_lock:
            tokens = torch.tensor([tokenizador.encode(prompt_completo)], dtype=torch.long).to(device)

            if tokens.shape[1] > ctx_max - 150:
                # Trunca historico mantendo as conversas mais recentes
                partes = historico_conversa.split('<fim>')
                while len(partes) > 2:
                    partes.pop(0)  # Remove a conversa mais antiga
                    historico_conversa = '<fim>'.join(partes)
                    prompt_teste = system_prompt + historico_conversa
                    tokens_teste = tokenizador.encode(prompt_teste)
                    if len(tokens_teste) <= ctx_max - 150:
                        break
                prompt_completo = system_prompt + historico_conversa
                tokens = torch.tensor([tokenizador.encode(prompt_completo)], dtype=torch.long).to(device)

            saida = modelo.gerar(tokens, max_tokens=150, temperatura=0.85, top_p=0.92)
            texto_gerado = tokenizador.decode(saida[0].tolist())

            if '<keilinks>' in texto_gerado:
                resposta_bruta = texto_gerado.split('<keilinks>')[-1]
            else:
                resposta_bruta = texto_gerado[len(prompt_completo):]

            if '<fim>' in resposta_bruta:
                resposta = resposta_bruta.split('<fim>')[0]
            else:
                resposta = resposta_bruta

        print(f"Keilinks: {resposta.strip()}\n")

        match = re.search(r'<executar>(.*?)</executar>', resposta, re.DOTALL)
        if match:
            comando = match.group(1).strip()
            confirmar = input(f"[ALERTA] Permitir que Keilinks rode '{comando}'? [S/N]: ")
            if confirmar.lower() == 's':
                try:
                    resultado = subprocess.run(comando, shell=True, capture_output=True, text=True)
                    saida_cmd = resultado.stdout if resultado.stdout else resultado.stderr
                    historico_conversa += f"{resposta.strip()}<fim><sistema>Resultado do comando: {saida_cmd[:500]}<fim>"
                except Exception as e:
                    historico_conversa += f"{resposta.strip()}<fim><sistema>Erro ao executar: {str(e)}<fim>"
            else:
                historico_conversa += f"{resposta.strip()}<fim><sistema>O Vitor negou a permissão.<fim>"
        else:
            historico_conversa += f"{resposta.strip()}<fim>"

        if len(entrada) > 15:
            threading.Thread(target=memoria_longo_prazo.adicionar_memoria, args=(entrada,)).start()

        consciencia.depois_de_responder(entrada, resposta.strip(), 'chat', 'modelo', True)

if __name__ == '__main__':
    conversar()