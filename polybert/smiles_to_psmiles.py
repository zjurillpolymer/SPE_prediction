import re
from rdkit import Chem

try:
    from canonicalize_psmiles import canonicalize
    _CANONICALIZE_AVAILABLE = True
except ImportError:
    _CANONICALIZE_AVAILABLE = False


def smiles_to_psmiles(smiles: str) -> str | None:
    """Convert polymer SMILES with metal end-groups to canonical PSMILES.

    Replaces [Cu] and [Au] end-group markers with [*] connection points
    (the PSMILES convention used by polyBERT), validates with RDKit,
    and optionally canonicalizes using the Ramprasad-Group library.
    """
    psmiles = smiles.replace('[Cu]', '[*]').replace('[Au]', '[*]')

    mol = Chem.MolFromSmiles(psmiles)
    if mol is None:
        return None

    if _CANONICALIZE_AVAILABLE:
        try:
            psmiles = canonicalize(psmiles)
        except Exception:
            pass

    return psmiles
