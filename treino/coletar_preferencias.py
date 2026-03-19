import json
import os

def coletar():
    arquivo = 'dados/rlhf_preferencias.jsonl'
    
    if not os.path.exists('dados'):
        os.makedirs('dados')
        
    while True:
        prompt = input("Vitor: ")
        if prompt.lower() == 'sair':
            break
            
        resposta_a = input("Insira Resposta A gerada: ")
        resposta_b = input("Insira Resposta B gerada: ")
        
        escolha = input("Qual soa mais natural? (A/B): ").strip().upper()
        
        if escolha in ['A', 'B']:
            dado = {
                "prompt": prompt,
                "resposta_aceita": resposta_a if escolha == 'A' else resposta_b,
                "resposta_rejeitada": resposta_b if escolha == 'A' else resposta_a
            }
            with open(arquivo, 'a', encoding='utf-8') as f:
                f.write(json.dumps(dado, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    coletar()