# Preparação local da Keilinks V4

Use um agente com acesso ao terminal dentro do WSL2/Ubuntu e peça que ele execute, valide e documente o roteiro de `KEILINKS_V4.md` sem iniciar o treino longo.

O agente deve obrigatoriamente:

1. Trabalhar na branch `keilinks-v4-training`, dentro de `/home/...`, nunca em `/mnt/c`.
2. Verificar `nvidia-smi`, RAM, disco, Python, PyTorch CUDA e BF16.
3. Executar `compileall` e o benchmark `treino.v4.benchmark_rtx5050`.
4. Gerar as conversas curadas e baixar primeiro um corpus pequeno.
5. Auditar idioma, spam, duplicatas, schemas e manifestos de licença.
6. Criar o mix SFT com sintéticos limitados a 25%.
7. Construir e inspecionar `dados/vocab_v4.json`.
8. Preparar binários de pré-treino e packing assistant-only.
9. Rodar somente smoke tests de 20 passos para pré-treino e SFT.
10. Avaliar o checkpoint com o conjunto congelado.
11. Testar busca web, URLs de fonte, cache, SSRF e autenticação administrativa.
12. Não enviar corpora, binários, ambientes virtuais ou checkpoints ao Git.
13. Não apagar dados/checkpoints existentes sem backup.
14. Entregar relatório com tokens/s, VRAM, tamanhos, tokens, fontes, licenças, rejeições, duplicatas, testes e comandos finais.

O prompt detalhado, com todas as regras e comandos, foi fornecido ao proprietário junto da entrega da PR #1.
