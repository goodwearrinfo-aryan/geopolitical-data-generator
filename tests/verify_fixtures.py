#!/usr/bin/env python3
"""Verify generated test fixtures."""

import pandas as pd


def main():
    # Check WDI
    gdp = pd.read_csv('tests/fixtures/world_bank/v1/gdp.csv')
    assert len(gdp) == 500, f'Expected 500 rows, got {len(gdp)}'
    assert gdp['gdp_usd'].min() > 0

    # Check INSCR
    polity = pd.read_csv('tests/fixtures/inscr/v1/polity.csv')
    assert len(polity) == 500
    assert polity['polity2'].between(-10, 10).all()

    # Check ACLED
    events = pd.read_csv('tests/fixtures/acled/v1/events.csv')
    assert len(events) > 10000
    assert events['fatalities'].min() >= 0

    print('All fixture checks passed!')


if __name__ == '__main__':
    main()