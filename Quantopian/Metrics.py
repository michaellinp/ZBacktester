import numpy as np
import pandas as pd
from zipline.api import get_datetime

from Quantopian.Logger import Logger

log = Logger()


class Metrics():
    def __init__(self, context):
        self.nominal_prices = pd.Series()
        self.prices = pd.Series()
        self.d_returns = pd.Series()
        self.rolling_annual_returns = pd.Series()
        self.rolling_n_day_SR = pd.Series()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def update(self, context, data):
        for s_no, s in enumerate(context.strategies):
            self._update_strategy_metrics(context, data, s, s_no)
            context.metrics.prices[get_datetime()] = np.sum([strat.metrics.prices for strat in context.strategies])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _update_strategy_metrics(self, context, data, s, s_no):
        ''' calculate and store current price of strategies used by algo '''
        price = self._update_portfolio_metrics(context, data, s) * context.strategy_weights[s_no]
        s.metrics.prices[get_datetime()] = price
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _update_portfolio_metrics(self, context, data, s):
        strategy_price = 0
        for p_no, p in enumerate(s.portfolios):
            price = np.sum([context.portfolio.positions[a].amount *
                            data.current(a, 'price') for a in p.all_assets]) * s.weights[p_no]
            p.metrics.prices[get_datetime()] = price
            strategy_price += price
            try:
                calculate_SR = context.calculate_SR
                nominal_price = np.sum([p.allocation_model.weights[i] *
                                        data.current(security, 'price') for i, security in enumerate(p.securities)])
                p.metrics.nominal_prices[get_datetime()] = nominal_price
            except:
                calculate_SR = False

            if calculate_SR:
                try:
                    context.SR_lookback = context.SR_lookback
                    context.SD_factor = context.SD_factor
                except:
                    raise ValueError('context.SR_lookback and/or context.SR_factor missing')
                    return

                p.metrics.rolling_n_day_SR[get_datetime()] = self._calculate_SR(context, data, p)

        return strategy_price

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _calculate_SR(self, context, data, p):
        rets = data.history(p.securities, 'price', context.SR_lookback, '1d').pct_change(1)
        portfolio_rets = (rets * p.allocation_model.weights).sum(axis=1)
        risk_free_rets = data.history(context.risk_free, 'price', context.SR_lookback, '1d').pct_change(1)
        excess_returns = portfolio_rets[1:] - risk_free_rets[1:]
        return excess_returns.mean() / portfolio_rets.std()