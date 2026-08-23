import torch

from tmol.database import ParameterDatabase
from tmol.score import ScoreType
from tmol.score.terms import TermCreator, score_term_creator


@score_term_creator
class SugarBBTermCreator(TermCreator):
    _score_types = [ScoreType.sugar_bb]

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        from tmol.score.sugar_bb import SugarBBEnergyTerm

        return SugarBBEnergyTerm(param_db, device)

    @classmethod
    def score_types(cls):
        return cls._score_types
