"""Compatibilidade: use o coletor V5 com licença e proveniência explícitas.

O coletor anterior tratava um corpus textual como pares instrução/resposta e
baixava fontes compostas sem registrar termos. A entrada V5 impede isso.
"""

from treino.v5.coletar_datasets import main


if __name__ == "__main__":
    main()
