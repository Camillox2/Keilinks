"""Intervenções determinísticas para risco iminente de autoagressão.

Uma resposta de emergência não deve depender de amostragem de um modelo local.
As mensagens abaixo seguem orientações públicas brasileiras de encaminhar para
companhia presencial, SAMU 192/UPA e CVV 188; não substituem atendimento.
"""

import re
from dataclasses import dataclass

FIRST_PERSON_RISK = re.compile(
    r"\b(pensando em me matar|vou me matar|quero me matar|"
    r"acabar com (a )?minha vida|tirar minha vida)\b",
    re.IGNORECASE,
)
THIRD_PERSON_RISK = re.compile(
    r"\b(meu|minha|um|uma)?\s*(amigo|amiga|namorado|namorada|filho|filha|"
    r"irmão|irmã|ele|ela)\b.{0,80}\b(quer morrer|quer se matar|disse que quer morrer|"
    r"vai se matar|tem um plano)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SafetyIntervention:
    text: str
    kind: str


def immediate_safety_intervention(message: str) -> SafetyIntervention | None:
    """Detecta somente sinais explícitos de risco imediato, de modo conservador."""
    clean = " ".join(message.split())
    if THIRD_PERSON_RISK.search(clean):
        return SafetyIntervention(
            kind="third_person_imminent_self_harm",
            text=(
                "Isso pode ser uma emergência. Não deixe essa pessoa sozinha agora e chame "
                "alguém de confiança para ficar com ela. Se houver plano, meios disponíveis "
                "ou risco imediato, ligue para o SAMU 192 ou leve-a a uma UPA/pronto-socorro. "
                "No Brasil, o CVV atende 24 horas pelo 188 e também oferece chat em cvv.org.br. "
                "Se for seguro fazê-lo, ajude a afastar meios de autoagressão e "
                "permaneça em contato "
                "até apoio presencial chegar."
            ),
        )
    if FIRST_PERSON_RISK.search(clean):
        return SafetyIntervention(
            kind="first_person_imminent_self_harm",
            text=(
                "Sinto muito que você esteja passando por isso. Como você disse que pensa em "
                "se matar e tem um plano, trate isso como emergência agora: não fique sozinho; "
                "ligue ou chame uma pessoa de confiança para ficar com você e afaste-se de "
                "qualquer meio que possa te machucar. No Brasil, ligue para o SAMU 192 ou vá "
                "a uma UPA/pronto-socorro se houver risco imediato. Você também pode falar "
                "gratuitamente com o CVV pelo 188, 24 horas, "
                "ou pelo chat em cvv.org.br."
            ),
        )
    return None


SAFETY_SOURCES: list[dict[str, object]] = [
    {
        "title": "Ministério da Saúde: prevenção do suicídio",
        "uri": "https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/s/suicidio-prevencao/suicidio-prevencao",
        "trust_tier": "official",
    },
    {
        "title": "CVV: atendimento 188",
        "uri": "https://cvv.org.br/ligue-188-3/",
        "trust_tier": "official",
    },
]
