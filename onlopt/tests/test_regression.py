"""回帰テスト: 固定シードでの RunResult のハッシュ比較(仕様 8.4)。

数値の意図しない変化を検出する。アルゴリズムを意図的に変更した場合は
期待値を更新すること(テスト失敗時に実際のハッシュが表示される)。
"""

import hashlib

import numpy as np

from onlopt import Hedge, Simulator, StochasticEnv, TsallisINF


def digest(result) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(result.losses).tobytes())
    h.update(np.ascontiguousarray(result.cum_regret).tobytes())
    if result.cum_pseudo_regret is not None:
        h.update(np.ascontiguousarray(result.cum_pseudo_regret).tobytes())
    return h.hexdigest()[:16]


EXPECTED_HEDGE = "d62097af31bbdb74"
EXPECTED_TSALLIS = "9cc7a683b7cc7bb4"


def test_hedge_fixed_seed_hash():
    res = Simulator().run(
        Hedge(5, horizon=1000), StochasticEnv.from_gap(5, 0.2), T=1000, seed=123
    )
    assert digest(res) == EXPECTED_HEDGE, f"hash changed: {digest(res)}"


def test_tsallis_fixed_seed_hash():
    res = Simulator().run(
        TsallisINF(5), StochasticEnv.from_gap(5, 0.2), T=1000, seed=123
    )
    assert digest(res) == EXPECTED_TSALLIS, f"hash changed: {digest(res)}"
