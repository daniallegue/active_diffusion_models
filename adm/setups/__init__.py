from .toy import ToyProblemSetup
from .corridor import CorridorProblemSetup
from .molecules import QM9ProblemSetup, GEOMDrugsProblemSetup
from .stable_diffusion import StableDiffusionProblemSetup


setups = {
    "toy": ToyProblemSetup,
    "corridor": CorridorProblemSetup,
    "qm9": QM9ProblemSetup,
    "geom_drugs": GEOMDrugsProblemSetup,
    "stable_diffusion": StableDiffusionProblemSetup,
}
