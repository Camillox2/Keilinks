"""
Treino de voz customizada com Piper TTS
Usa os clips processados pra criar a voz da Keilinks.

Pré-requisitos:
  pip install piper-tts piper-phonemize

Pipeline:
  1. Converte metadata.csv + clips WAV pro formato do Piper
  2. Treina modelo VITS single-speaker
  3. Exporta modelo .onnx pra inferência rápida

Uso:
  python voz/treinar_voz.py
  python voz/treinar_voz.py --epocas 2000
  python voz/treinar_voz.py --testar  (testa modelo treinado)
"""

import os
import sys
import json
import argparse
import subprocess

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def verificar_dados():
    """Verifica se os dados de treino estão prontos"""
    clips_dir = 'voz/clips'
    metadata = 'voz/metadata.csv'

    if not os.path.exists(clips_dir):
        print("ERRO: diretório voz/clips/ não encontrado")
        print("Execute primeiro: python voz/processar_audio.py --audio <arquivo>")
        return False

    if not os.path.exists(metadata):
        print("ERRO: voz/metadata.csv não encontrado")
        print("Execute primeiro: python voz/processar_audio.py --audio <arquivo>")
        return False

    # Conta clips
    wavs = [f for f in os.listdir(clips_dir) if f.endswith('.wav')]
    if len(wavs) < 50:
        print(f"AVISO: apenas {len(wavs)} clips. Recomendado: 150+ pra boa qualidade")

    # Conta linhas no metadata
    with open(metadata, 'r', encoding='utf-8') as f:
        linhas = [l.strip() for l in f if l.strip()]

    print(f"  Clips WAV: {len(wavs)}")
    print(f"  Entradas metadata: {len(linhas)}")
    return True


def criar_config_piper():
    """Cria configuração de treino do Piper"""
    config = {
        "audio": {
            "sample_rate": 22050,
            "mel_channels": 80,
            "hop_length": 256,
            "win_length": 1024,
            "fmin": 0,
            "fmax": 8000
        },
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "num_languages": 1,
            "num_speakers": 1,
            "use_sdp": True
        },
        "training": {
            "seed": 42,
            "epochs": 2000,
            "learning_rate": 2e-4,
            "betas": [0.8, 0.99],
            "eps": 1e-9,
            "batch_size": 16,
            "fp16_run": True,
            "segment_size": 8192
        },
        "language": {
            "code": "pt-br",
            "phoneme_type": "espeak"
        },
        "dataset": {
            "path": "voz/clips",
            "metadata": "voz/metadata.csv",
            "delimiter": "|"
        }
    }

    config_path = 'voz/config_treino.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  Config salva em: {config_path}")
    return config_path


def treinar(epocas=2000):
    """Treina o modelo de voz com Piper"""
    print("=" * 60)
    print("  Treino de Voz — Keilinks")
    print("=" * 60)

    if not verificar_dados():
        return

    config_path = criar_config_piper()

    print(f"\n  Épocas: {epocas}")
    print(f"  Config: {config_path}")

    # Verifica se piper-tts está instalado
    try:
        import piper
        print(f"  Piper TTS: instalado")
    except ImportError:
        print("\n  Piper TTS não instalado. Para instalar:")
        print("    pip install piper-tts")
        print("\n  Alternativa: usar o treinador standalone do Piper:")
        print("    git clone https://github.com/rhasspy/piper.git")
        print("    cd piper/src/python")
        print("    pip install -e .")
        print(f"\n  Depois execute o treino manualmente:")
        print(f"    python -m piper_train \\")
        print(f"      --dataset-dir voz/clips \\")
        print(f"      --accelerator gpu \\")
        print(f"      --batch-size 16 \\")
        print(f"      --max-epochs {epocas} \\")
        print(f"      --checkpoint-epochs 100 \\")
        print(f"      --quality medium")
        return

    # Treino via linha de comando do Piper
    os.makedirs('voz/modelo', exist_ok=True)

    cmd = [
        sys.executable, '-m', 'piper_train',
        '--dataset-dir', 'voz/clips',
        '--accelerator', 'gpu',
        '--devices', '1',
        '--batch-size', '16',
        '--validation-split', '0.1',
        '--max-epochs', str(epocas),
        '--checkpoint-epochs', '100',
        '--quality', 'medium',
        '--default-root-dir', 'voz/modelo',
    ]

    print(f"\n  Iniciando treino...")
    print(f"  Comando: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n  Treino concluído!")
        print(f"  Modelo salvo em: voz/modelo/")
        exportar_onnx()
    except subprocess.CalledProcessError as e:
        print(f"\n  ERRO no treino: {e}")
    except FileNotFoundError:
        print(f"\n  ERRO: piper_train não encontrado")
        print(f"  Instale: pip install piper-tts")


def exportar_onnx():
    """Exporta modelo treinado para ONNX (inferência rápida)"""
    print("\n  Exportando pra ONNX...")

    # Procura o último checkpoint
    modelo_dir = 'voz/modelo'
    ckpts = []
    for root, dirs, files in os.walk(modelo_dir):
        for f in files:
            if f.endswith('.ckpt'):
                ckpts.append(os.path.join(root, f))

    if not ckpts:
        print("  Nenhum checkpoint encontrado pra exportar")
        return

    ultimo = sorted(ckpts)[-1]
    saida = 'voz/keilinks_voz.onnx'

    cmd = [
        sys.executable, '-m', 'piper_train.export_onnx',
        ultimo, saida
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"  Modelo ONNX exportado: {saida}")

        # Cria config JSON pro modelo
        config_modelo = {
            "language": "pt-br",
            "phoneme_type": "espeak",
            "sample_rate": 22050,
            "speaker": "keilinks"
        }
        config_path = saida + '.json'
        with open(config_path, 'w') as f:
            json.dump(config_modelo, f, indent=2)

        print(f"  Config do modelo: {config_path}")
        print(f"\n  Pra testar: python voz/treinar_voz.py --testar")
    except Exception as e:
        print(f"  ERRO na exportação: {e}")


def testar():
    """Testa o modelo de voz treinado"""
    modelo_path = 'voz/keilinks_voz.onnx'

    if not os.path.exists(modelo_path):
        print("ERRO: modelo não encontrado. Treine primeiro:")
        print("  python voz/treinar_voz.py")
        return

    try:
        from piper import PiperVoice
    except ImportError:
        print("ERRO: piper-tts não instalado")
        print("  pip install piper-tts")
        return

    print("=" * 60)
    print("  Teste de Voz — Keilinks")
    print("=" * 60)

    voice = PiperVoice.load(modelo_path)

    frases_teste = [
        "Oi! Eu sou a Keilinks, prazer em te conhecer.",
        "Posso te ajudar com programação, tecnologia e muito mais.",
        "Que legal! Me conta mais sobre o seu projeto.",
    ]

    os.makedirs('voz/teste', exist_ok=True)

    for i, frase in enumerate(frases_teste):
        saida = f'voz/teste/teste_{i}.wav'
        with open(saida, 'wb') as f:
            voice.synthesize(frase, f)
        print(f"  [{i+1}] \"{frase}\"")
        print(f"      -> {saida}")

    print(f"\n  Áudios de teste salvos em voz/teste/")
    print(f"  Ouça e avalie a qualidade!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treino de voz Keilinks')
    parser.add_argument('--epocas', type=int, default=2000, help='Épocas de treino (default: 2000)')
    parser.add_argument('--testar', action='store_true', help='Testa modelo treinado')
    parser.add_argument('--exportar', action='store_true', help='Exporta checkpoint pra ONNX')
    args = parser.parse_args()

    if args.testar:
        testar()
    elif args.exportar:
        exportar_onnx()
    else:
        treinar(epocas=args.epocas)
