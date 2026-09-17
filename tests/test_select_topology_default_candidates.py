"""The default topology race builds each candidate with its constructor's own
defaults (#2899 P10). ``select_topology`` and ``TopologyAutoSelector("torus")``
raced a 12 x 12 torus while ``topology.Torus()`` builds 20 x 20, so the automatic
race priced a different torus than the one a user gets by name."""

from gamfit import _select_topology as st
from gamfit import topology


def test_default_tensor_candidates_use_their_constructor_defaults():
    for name, constructor in (("torus", topology.Torus), ("cylinder", topology.Cylinder)):
        candidate = st._default_topology_candidate(name, 2)
        assert candidate.name == name
        assert candidate.topology._gamfit_tensor_k == constructor()._gamfit_tensor_k
