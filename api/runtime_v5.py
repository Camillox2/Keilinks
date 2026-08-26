"""Compatibilidade V5 para o runtime V6 seguro.

O runtime anterior carregava um checkpoint V4 aleatório, executava busca web
sem controle e persistia conversas sem consentimento. O caminho suportado usa
o adaptador QLoRA auditado e o runtime em ``keilinks_v5``.
"""

from __future__ import annotations

from keilinks_v5.runtime import ChatAnswer as RuntimeAnswerV5
from keilinks_v5.runtime import ChatMessage, UnslothRuntime
from keilinks_v5.settings import KeilinksSettings


class V5Runtime(UnslothRuntime):
    """Alias de migração; inicialize com ``KeilinksSettings``.

    Exemplo: ``V5Runtime(KeilinksSettings.from_env(), store)``. Não há suporte
    para o antigo par ``checkpoint_path``/``vocab_path`` porque ele pertence a
    uma arquitetura incompatível e não representa o produto atual.
    """


__all__ = ["ChatMessage", "KeilinksSettings", "RuntimeAnswerV5", "V5Runtime"]
