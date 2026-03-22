"""
Processamento de áudio para treino de voz da Keilinks
Recebe gravações brutas e prepara para o Piper TTS.

Pipeline:
  1. Corta silêncios longos pra separar frases
  2. Normaliza volume
  3. Exporta clips individuais em WAV 22050Hz mono
  4. Gera metadata.csv no formato do Piper

Dependências:
  pip install pydub

Uso:
  python voz/processar_audio.py --audio voz/gravacao.wav
  python voz/processar_audio.py --audio voz/gravacao.wav --min-silencio 500
"""

import os
import csv
import argparse
import re

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
except ImportError:
    print("ERRO: pydub não instalado. Execute:")
    print("  pip install pydub")
    print("\nTambém precisa do ffmpeg:")
    print("  Windows: baixe de https://ffmpeg.org/download.html e adicione ao PATH")
    exit(1)


def carregar_frases():
    """Carrega as frases do arquivo de gravação"""
    caminho = 'voz/frases_gravacao.txt'
    if not os.path.exists(caminho):
        print("AVISO: frases_gravacao.txt não encontrado.")
        print("Execute: python voz/gerar_frases.py")
        return []

    frases = []
    with open(caminho, 'r', encoding='utf-8') as f:
        for linha in f:
            # Formato: "  123. Texto da frase"
            m = re.match(r'\s+\d+\.\s+(.+)', linha)
            if m:
                frases.append(m.group(1).strip())
    return frases


def processar(caminho_audio, min_silencio=600, limiar_silencio=-40):
    """
    Processa áudio bruto em clips individuais

    Args:
        caminho_audio: caminho do WAV/MP3 gravado
        min_silencio: mínimo de silêncio (ms) pra considerar separação
        limiar_silencio: volume abaixo do qual é silêncio (dBFS)
    """
    if not os.path.exists(caminho_audio):
        print(f"ERRO: arquivo não encontrado: {caminho_audio}")
        return

    print("=" * 60)
    print("  Processamento de Áudio — Keilinks Voice")
    print("=" * 60)

    # Carrega frases esperadas
    frases = carregar_frases()
    print(f"\n  Frases esperadas: {len(frases)}")

    # Carrega áudio
    print(f"  Carregando: {caminho_audio}")
    ext = os.path.splitext(caminho_audio)[1].lower()
    if ext == '.mp3':
        audio = AudioSegment.from_mp3(caminho_audio)
    elif ext == '.wav':
        audio = AudioSegment.from_wav(caminho_audio)
    elif ext == '.ogg':
        audio = AudioSegment.from_ogg(caminho_audio)
    else:
        audio = AudioSegment.from_file(caminho_audio)

    duracao = len(audio) / 1000
    print(f"  Duração: {duracao:.1f}s ({duracao/60:.1f} min)")
    print(f"  Sample rate: {audio.frame_rate}Hz")
    print(f"  Canais: {audio.channels}")

    # Converte pra mono se necessário
    if audio.channels > 1:
        print("  Convertendo pra mono...")
        audio = audio.set_channels(1)

    # Normaliza volume
    print("  Normalizando volume...")
    target_dbfs = -20.0
    change = target_dbfs - audio.dBFS
    audio = audio.apply_gain(change)

    # Corta em clips por silêncio
    print(f"  Cortando por silêncio (min={min_silencio}ms, limiar={limiar_silencio}dBFS)...")
    clips = split_on_silence(
        audio,
        min_silence_len=min_silencio,
        silence_thresh=limiar_silencio,
        keep_silence=200  # mantém 200ms de silêncio nas bordas
    )

    print(f"  Clips detectados: {len(clips)}")

    if len(clips) == 0:
        print("\n  ERRO: Nenhum clip detectado. Tente:")
        print("    --min-silencio 400  (reduz mínimo de silêncio)")
        print("    Ou verifique se o áudio não está muito baixo")
        return

    # Cria diretório de saída
    saida_dir = 'voz/clips'
    os.makedirs(saida_dir, exist_ok=True)

    # Resample pra 22050Hz (padrão do Piper)
    taxa_alvo = 22050

    # Exporta clips e gera metadata
    metadata = []
    clips_curtos = 0
    clips_longos = 0

    for i, clip in enumerate(clips):
        duracao_clip = len(clip) / 1000

        # Filtra clips muito curtos (< 0.5s = provavelmente ruído)
        if duracao_clip < 0.5:
            clips_curtos += 1
            continue

        # Filtra clips muito longos (> 15s = provavelmente erro de corte)
        if duracao_clip > 15:
            clips_longos += 1
            # Ainda salva, mas avisa
            print(f"  AVISO: clip {i} muito longo ({duracao_clip:.1f}s) — verifique manualmente")

        # Resample
        clip = clip.set_frame_rate(taxa_alvo)

        # Nome do arquivo
        nome = f"kei_{i:04d}.wav"
        caminho_clip = os.path.join(saida_dir, nome)

        # Exporta WAV 22050Hz mono 16bit
        clip.export(caminho_clip, format="wav", codec="pcm_s16le")

        # Associa com frase (se disponível)
        texto = frases[i] if i < len(frases) else f"[FRASE {i+1} - PREENCHER MANUALMENTE]"
        metadata.append((nome, texto))

    # Salva metadata.csv (formato Piper)
    csv_path = 'voz/metadata.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        for nome, texto in metadata:
            # Formato Piper: nome_sem_ext|texto|texto_normalizado
            base = os.path.splitext(nome)[0]
            writer.writerow([base, texto, texto])

    print(f"\n  Resultado:")
    print(f"    Clips válidos:   {len(metadata)}")
    print(f"    Clips curtos:    {clips_curtos} (descartados, < 0.5s)")
    if clips_longos:
        print(f"    Clips longos:    {clips_longos} (mantidos, verificar)")
    print(f"    Salvos em:       {saida_dir}/")
    print(f"    Metadata:        {csv_path}")
    print(f"    Sample rate:     {taxa_alvo}Hz")

    if len(metadata) != len(frases) and frases:
        print(f"\n  ATENÇÃO: {len(metadata)} clips vs {len(frases)} frases esperadas")
        print(f"  Verifique metadata.csv e ajuste as frases manualmente se necessário")
        print(f"  (clips podem ter sido divididos ou unidos incorretamente)")

    print(f"\n  Próximo passo:")
    print(f"    1. Verifique os clips em {saida_dir}/")
    print(f"    2. Corrija metadata.csv se alguma frase não bater")
    print(f"    3. Execute: python voz/treinar_voz.py")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processa áudio para Piper TTS')
    parser.add_argument('--audio', required=True, help='Arquivo de áudio (WAV, MP3, OGG)')
    parser.add_argument('--min-silencio', type=int, default=600, help='Silêncio mínimo em ms (default: 600)')
    parser.add_argument('--limiar', type=int, default=-40, help='Limiar de silêncio em dBFS (default: -40)')
    args = parser.parse_args()
    processar(args.audio, min_silencio=args.min_silencio, limiar_silencio=args.limiar)
