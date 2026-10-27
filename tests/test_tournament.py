import numpy as np
import pytest

from mmsim.bots import SymmetricQuoter
from mmsim.flow import CSTParams, ZeroIntelligence
from mmsim.tournament import session, standings

SKY = CSTParams.sky_perfect()


def flow():
    return ZeroIntelligence(SKY)


ENTRANTS = {"a": lambda: SymmetricQuoter(0.5), "b": lambda: SymmetricQuoter(0.5, keep=1)}


def test_session_is_deterministic():
    one = session({"a": ENTRANTS["a"]}, 4, flow=flow, minutes=60)
    two = session({"a": ENTRANTS["a"]}, 4, flow=flow, minutes=60)
    assert one == two


def test_session_marks_everyone_at_the_same_close():
    res = session(ENTRANTS, 1, flow=flow, minutes=120)
    assert set(res) == {"a", "b"}
    assert all(r.fills > 0 for r in res.values())


def test_attribution_adds_up_to_pnl():
    res = session(ENTRANTS, 2, flow=flow, minutes=120)
    for r in res.values():
        assert r.spread + r.adverse + r.inventory == pytest.approx(r.pnl)


def test_same_quoters_split_the_flow():
    together = [session(ENTRANTS, s, flow=flow, minutes=120) for s in range(6)]
    alone = [session({"a": ENTRANTS["a"]}, s, flow=flow, minutes=120) for s in range(6)]
    solo_fills = np.mean([r["a"].fills for r in alone])
    shared = np.mean([r["a"].fills + r["b"].fills for r in together])
    assert shared < 2 * solo_fills


def test_standings_wins_sum_to_one_and_sort_by_mean():
    together = [session(ENTRANTS, s, flow=flow, minutes=60) for s in range(5)]
    alone = [
        {n: session({n: f}, s, flow=flow, minutes=60)[n] for n, f in ENTRANTS.items()}
        for s in range(5)
    ]
    rows = standings(together, alone)
    assert sum(r.wins for r in rows) == pytest.approx(1.0)
    assert rows[0].mean >= rows[1].mean
    for r in rows:
        assert r.crowding[0] == pytest.approx(np.mean(r.pnl - r.solo))
