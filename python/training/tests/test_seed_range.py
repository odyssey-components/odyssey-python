from odylab_training.seed_range import SeedRange


def test_half_open_seed_intervals():
    seeds = SeedRange(10, 5)
    assert 10 in seeds
    assert 15 not in seeds
    assert not seeds.overlaps(SeedRange(15, 2))
    assert seeds.overlaps(SeedRange(14, 2))
    assert not seeds.overlaps(SeedRange(12, 0))
