import datetime as dt
import pytz

import numpy as np
import pandas as pd
import math
import talib
import re
import collections
from cvxopt import matrix, solvers, spdiag

from zipline import TradingAlgorithm, run_algorithm
from zipline.api import order, order_target_percent, get_open_orders, get_datetime, record, schedule_function
from zipline.api import history, get_environment, sid, set_commission, set_slippage, symbol, symbols
from zipline.api import date_rules, time_rules, commission, slippage, set_symbol_lookup_date

from Quantopian.Data import Data
from Quantopian.Metrics import Metrics

GTC_LIMIT = 10

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dummy Logger
class Logger():
    pass

    def info(self, s):
        print('{} INFO : {}'.format(get_datetime().tz_convert('US/Eastern'), s))
        pass

    def debug(self, s):
        print('{} DEBUG : {}'.format(get_datetime().tz_convert('US/Eastern'), s))
        pass

    def warn(self, s):
        print('{} WARNING : {}'.format(get_datetime().tz_convert('US/Eastern'), s))
        pass

log = Logger()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Algo():
    def __init__(self, context, strategies=[], allocation_model=None, regime=None):

        if get_environment('platform') == 'zipline':
            self.day_no = 0

        self.ID = 'algo'

        self.strategies = strategies
        self.allocation_model = allocation_model
        self.regime = regime

        context.strategies = self.strategies
        self.weights = [0. for s in self.strategies]
        context.strategy_weights = self.weights
        self.strategy_IDs = [s.ID for s in self.strategies]
        self.active = [s.ID for s in self.strategies] + [p.ID for s in self.strategies for p in s.portfolios]

        if self.allocation_model == None:
            raise ValueError('\n *** FATAL ERROR : ALGO ALLOCATION MODEL CANNOT BE NONE ***\n')

        context.metrics = Metrics(context)

        self.all_assets = self._set_all_assets()

        print('ALL ASSETS = {}'.format([s.symbol for s in self.all_assets]))

        self.allocations = pd.Series(0, index=self.all_assets)
        self.previous_allocations = pd.Series(0, index=self.all_assets)

        self.data = Data(self.all_assets)
        context.algo_data = self.data

        set_symbol_lookup_date('2016-01-01')

        self._instantiate_rules(context)

        context.securities = []  # placeholder securities in portfolio

        if get_environment('platform') == 'zipline':
            context.count = context.max_lookback
        else:
            context.count = 0

        self.rebalance_count = 1  # default rebalance interval = 1
        self.first_time = True

        return
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _instantiate_rules(self, context):
        context.rules = {}
        for r in context.algo_rules:
            context.rules[r.name] = r
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _set_all_assets(self):
        all_assets = [s.all_assets for s in self.strategies]
        self.all_assets = set([i for sublist in all_assets for i in sublist])
        return self.all_assets

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def update_data(self, context, data):

        if get_environment('platform') == 'zipline':
            # allow data buffer to fill in the ZIPLINE ENVIRONMENT
            if self.day_no <= context.max_lookback:
                self.day_no += 1
                return

        context.algo_data = self.data.update(context, data)

        # print ('ALGO_DATA = {}'.format(context.algo_data))

        return context.algo_data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _allocate_assets(self, context):
        log.debug('STRATEGY WEIGHTS = {}\n'.format(self.weights))
        for i, s in enumerate(self.strategies):
            self.allocations = self.allocations.add(self.weights[i] * s.allocations,
                                                    fill_value=0)
        if 1. - sum(self.allocations) > 1.e-15:
            raise RuntimeError('SUM OF ALLOCATIONS = {} - SHOULD ALWAYS BE 1'.format(sum(self.allocations)))

        return self.allocations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def rebalance(self, context, data):

        # make sure there's algo data
        if not isinstance(context.algo_data, dict):
            return
        elif not self.first_time:
            if self.rebalance_count != context.rebalance_interval:
                self.rebalance_count += 1
                return

        self.first_time = False

        self.rebalance_count = 1

        log.info('----------------------------------------------------------------------------')
        log.debug(get_datetime())

        self.allocations = pd.Series(0., index=self.all_assets)
        self.elligible = pd.Index(self.strategy_IDs)

        self.allocation_model.caller = self
        if self.regime == None:
            self._get_strategy_and_portfolio_allocations(context)
        else:
            self._check_for_regime_change_and_set_active(context)

        self.weights = self.allocation_model.get_weights(context)
        self.allocations = self._allocate_assets(context)

        self._execute_orders(context, data)

        return self.allocations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _get_strategy_and_portfolio_allocations(self, context):
        for s_no, s in enumerate(self.strategies):
            s.allocations = pd.Series(0., index=s.all_assets)
            for p_no, p in enumerate(s.portfolios):
                p.allocations = pd.Series(0., index=p.all_assets)
                p.allocations = p.reallocate(context)
            s.allocations = s.reallocate(context)
        return

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_for_regime_change_and_set_active(self, context):
        self.current_regime = self.regime.get_current(context)
        log.debug('REGIME : {} \n'.format(self.current_regime))
        if self.regime.detect_change(context):
            self.regime.set_new_regime()
            self.active = self.regime.get_active()
        else:
            log.info('REGIME UNCHANGED. JUST REBALANCE\n')
        return
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _execute_orders(self, context, data):

        for s in self.allocations.index:
            if context.portfolio.positions[s].amount > 0 and self.allocations[s] == 0:
                order_target_percent(s, 0)
            elif self.allocations[s] != 0:
                if get_open_orders(s):
                    continue

                current_value = context.portfolio.positions[s].amount * data.current(s, 'price')
                portfolio_value = context.portfolio.portfolio_value
                if portfolio_value == 0:  # before first purchases
                    portfolio_value = context.account.available_funds
                target_value = portfolio_value * self.allocations[s]

                if np.abs(target_value / current_value - 1) < context.threshold:
                    continue

                order_target_percent(s, self.allocations[s] * context.leverage)
                qty = int(context.account.net_liquidation * self.allocations[s] / data.current(s, 'price'))
                log.debug('ORDERING {} : {}%  QTY = {}'.format(s.symbol, self.allocations[s] * 100, qty))

        context.gtc_count = GTC_LIMIT

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def check_for_unfilled_orders(self, context, data):
        unfilled = {o.sid: o.amount - o.filled for oo in get_open_orders() for o in get_open_orders(oo)}
        context.outstanding = {u: unfilled[u] for u in unfilled if unfilled[u] != 0}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def fill_outstanding_orders(self, context, data):
        if context.outstanding == {}:
            context.show_positions = False
            return
        elif context.gtc_count > 0:
            for s in context.outstanding:
                order(s, context.outstanding[s])
                log.debug('ORDER {} OUTSTANDING {} SHARES'.format(context.outstanding[s], s.symbol))

            context.gtc_count -= 1
        else:
            log.info('GTC_COUNT EXPIRED')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def update_metrics(self, context, data):
        tmp = context.metrics.update(context, data)
        if tmp != None:
            context.metrics = tmp
            return context.metrics
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def show_records(self, context, data):
        record('LEVERAGE', context.account.leverage)
        # record('CONTEXT_LEVERAGE', context.leverage)
        # record('DPF', context.dpf)
        # record('PV', context.account.total_positions_value)
        # record('PV1',context.portfolio.positions_value)
        # record('TOTAL', context.portfolio.portfolio_value)
        # record('CASH', context.portfolio.cash)
        # record('ALGO_PRICE', context.metrics.prices[-1])
        # for s in context.strategies:
        #     # record(s.ID + '_price', s.metrics.prices[-1])
        #     for p in s.portfolios:
        #         record(p.ID + '_price', p.metrics.prices[-1])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def show_positions(self, context, data):

        log.info('\nPOSITIONS\n')
        for asset in self.all_assets:
            if context.portfolio.positions[asset].amount > 0:
                log.info(
                    '{0} : QTY = {1}, COST BASIS {2:3.2f}, CASH = {3:7.2f}, POSITIONS VALUE = {4:7.2f}, TOTAL = {5:7.2f}'
                    .format(asset.symbol, context.portfolio.positions[asset].amount,
                            context.portfolio.positions[asset].cost_basis,
                            context.portfolio.cash,
                            context.portfolio.positions[asset].amount * data.current(asset, 'price'),
                            context.portfolio.portfolio_value))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Regime():
    def __init__(self, transitions):
        """Initialize Regime object. Set init state and transition table."""
        self.transitions = transitions
        # set current != new to always detect change on first reallocation
        self.current_regime = 0
        self.new_regime = 1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def detect_change(self, context):
        self.new_regime = self.get_current(context)
        return [False if self.current_regime == self.new_regime else True][0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_current(self, context):
        for k in self.transitions.keys():
            rule_name = self.transitions[k][0]
            rule = context.rules[rule_name]
            if rule.apply_rule(context):
                return k
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_new_regime(self):
        self.current_regime = self.new_regime
        record('REGIME', self.current_regime)
        log.info('REGIME CHANGE - NEW REGIME = {}'.format(self.current_regime))
        return self.current_regime
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_active(self):
        return self.transitions[self.current_regime][1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Strategy():
    def __init__(self, context, ID='', portfolios=[], allocation_model=None):

        self.ID = ID
        self.portfolios = portfolios
        self.portfolio_IDs = [p.ID for p in self.portfolios]
        self.weights = [0. for p in portfolios]

        self.metrics = Metrics(context)

        if allocation_model == None:
            self.allocation_model = AllocationModel(context, mode='EW')
        else:
            self.allocation_model = allocation_model

        self._set_all_assets()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _set_all_assets(self):
        all_assets = [p.all_assets for p in self.portfolios]
        self.all_assets = set([i for sublist in all_assets for i in sublist])
        return self.all_assets
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def allocate_assets(self):
        self.allocations = pd.Series(0., index=self.all_assets)
        log.debug('STRATEGY {} PORTFOLIO WEIGHTS = {}\n'.format(self.ID, [round(w, 2) for w in self.weights]))
        for i, p in enumerate(self.portfolios):
            self.allocations = self.allocations.add(self.weights[i] * p.allocations,
                                                    fill_value=0)
        log.debug('SECURITY ALLOCATIONS for {} \n{}\n'.format(self.ID, self.allocations.round(2)))
        return self.allocations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def reallocate(self, context):
        self.elligible = pd.Index(self.portfolio_IDs)
        self.allocation_model.caller = self
        self.weights = self.allocation_model.get_weights(context)
        self.allocations = self.allocate_assets()
        return self.allocations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Portfolio():
    def __init__(self, context, ID='',
                 securities=[], allocation_model=None,
                 scoring_model=None,
                 downside_protection_model=None,
                 cash_proxy=None, allow_shorts=False):

        self.ID = ID
        self.securities = securities
        self.weights = [0. for s in securities]
        self.allocation_model = allocation_model
        self.scoring_model = scoring_model
        self.score = scoring_model
        self.downside_protection_model = downside_protection_model
        if cash_proxy == None:
            log.info('NO CASH_PROXY SPECIFIED FOR PORTFOLIO {}'.format(self.ID))
            raise ValueError('INITIALIZATION ERROR')
        self.cash_proxy = cash_proxy

        self.metrics = Metrics(context)

        for s in [context.market_proxy, self.cash_proxy, context.risk_free]:
            if s in self.securities:
                log.warn('WARNING : {} is included in the portfolio'.format(s.symbol))

        self.all_assets = list(set(self.securities + [context.market_proxy, self.cash_proxy, context.risk_free]))

        self.allocations = pd.Series([0.0] * len(self.all_assets), index=self.all_assets)

        return

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def reallocate(self, context):

        self.allocations = pd.Series(0., index=self.all_assets)
        self.elligible = pd.Index(self.securities)

        if self.scoring_model != None and self.scoring_model.method != None:
            context.securities = self.securities[:]
            self.score = self.scoring_model.compute_score(context)
            self.elligible = self.scoring_model.apply_ntop()

        self.allocation_model.caller = self
        self.weights = self.allocation_model.get_weights(context)
        self.allocations[self.elligible] = self.weights

        log.debug('ALLOCATIONS FOR {} : {}\n'.format(self.ID,
                                                     [(self.allocations.index[i].symbol, round(v, 2))
                                                      for i, v in enumerate(self.allocations)
                                                      if v > 0]))

        if self.downside_protection_model != None:
            self.allocations = self.downside_protection_model.apply_protection(context,
                                                                               self.allocations,
                                                                               self.cash_proxy,
                                                                               [self.securities, self.score])
            log.debug('AFTER DOWNSIDE PROTECTION {} : {}\n'.format(self.ID,
                                                                   [(self.allocations.index[i].symbol, round(v, 2))
                                                                    for i, v in enumerate(self.allocations)
                                                                    if v > 0]))

        return self.allocations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class ScoringModel():
    def __init__(self, context, factors=None, method=None, n_top=1):
        self.factors = factors
        self.method = method
        if self.factors == None:
            raise ValueError('Unable to score model with no factors')
        # if self.method == None :
        #     raise ValueError ('Unable to score model with no method')
        self.n_top = n_top
        self.score = 0
        self.methods = {'RS': self._relative_strength,
                        'EAA': self._eaa
                        }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def compute_score(self, context):
        self.securities = context.securities
        self.score = self.methods[self.method](context)
        # log.debug ('\nSCORE\n\n{}\n'.format(self.score))
        return self.score

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _relative_strength(self, context):
        self.score = 0.
        for name in self.factors.keys():

            if np.isnan(context.algo_data[name[1:]][self.securities]).any():
                security = [(self.securities[s].symbol, v)
                            for s, v in enumerate(context.algo_data[name[1:]][self.securities]) if np.isnan(v)][0][0]
                raise RuntimeError('SCORING ERROR : FACTOR {} VALUE FOR {} IS nan'.format(name, security))

            if name[0] == '+':
                # log.debug('Values for factor {} :\n\{}\nRANKS : \n{}'.format(name[1:],
                #                                                              [(s.symbol, context.algo_data[name[1:]][s]) for s in self.securities],
                #                                                              [(s.symbol, context.algo_data[name[1:]][self.securities].rank(ascending=False)[s])
                #                                                               for s in self.securities]))

                try:
                    # highest value gets highest rank / score
                    self.score = self.score + context.algo_data[name[1:]][self.securities].rank(ascending=True) \
                                              * self.factors[name]
                except:
                    raise RuntimeError(
                        '\n *** FATAL ERROR : UNABLE TO SCORE FACTOR {}. CHECK TRANSFORM & FACTOR DEFINITIONS\n'
                        .format(name[1:]))

            elif name[0] == '-':
                # log.debug('Values for factor {} :\n\{}\nRANKS : \n{}'.format(name[1:],
                #                                                              [(s.symbol, context.algo_data[name[1:]][s]) for s in self.securities],
                #                                                              [(s.symbol, context.algo_data[name[1:]][self.securities].rank(ascending=True)[s])
                #                                                               for s in self.securities]))

                try:
                    # lowest value gets highest rank /score
                    self.score = self.score + context.algo_data[name[1:]][self.securities].rank(ascending=False) \
                                              * self.factors[name]
                except:
                    raise RuntimeError('\n UNABLE TO SCORE FACTOR {}. CHECK TRANSFORM & FACTOR DEFINITIONS\n'
                                       .format(name[1:]))

        # log.debug('Scores for factor {} :\n\n{}'.format(name[1:],
        #                                                 [(s.symbol, self.score[s]) for s in self.securities]))

        return self.score
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _eaa(self, context):

        # prices = data.history(self.securities, 'price', 280, '1d')
        prices = context.raw_data['price'][self.securities]

        monthly_prices = prices.resample('M')[self.securities]
        monthly_returns = monthly_prices.pct_change().ix[-12:]

        # nominal return correlation to equi-weight portfolio
        N = len(self.securities)
        equal_weighted_index = monthly_returns.mean(axis=1)
        C = pd.Series([0.0] * N, index=self.securities)
        for s in C.index:
            C[s] = monthly_returns[s].corr(equal_weighted_index)

        R = context.algo_data['R'][self.securities]
        V = monthly_returns.std()

        # Apply factor weights
        # wi ~ zi = ( ri^wR * (1-ci)^wC / vi^wV )^wS
        wR = self.factors['R']
        wC = self.factors['C']
        wV = self.factors['V']
        wS = self.factors['S']
        eps = self.factors['eps']

        # Generalized Momentum Score
        self.score = ((R ** wR) * ((1 - C) ** wC) / (V ** wV)) ** (wS + eps)

        return self.score

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def apply_ntop(self):

        N = len(self.securities)
        if self.method == 'EAA':
            self.n_top = min(np.ceil(N ** 0.5) + 1, N / 2)
            elligible = self.score.sort_values().index[-self.n_top:]
        else:
            # best score gets lowest rank
            ranks = self.score.rank(ascending=False, method='dense')
            elligible = ranks[ranks <= self.n_top].index

        return elligible
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class AllocationModel():
    def __init__(self, context, mode='EW', weights=None, rule=None, formula=None, kwargs={}):
        self.mode = mode
        self.formula = formula
        self.weights = weights
        self.rule = rule
        self.kwargs = kwargs

        self.modes = {'EW': self._equal_weight_allocation,
                      'FIXED': self._fixed_allocation,
                      'PROPORTIONAL': self._proportional_allocation,
                      'MIN_VARIANCE': self._min_variance_allocation,
                      'MAX_SHARPE': self._max_sharpe_allocation,
                      'NOMINAL_SHARPE': self._nominal_sharpe_allocation,
                      'BY_FORMULA': self._allocation_by_formula,
                      'REGIME_EW': self.allocate_by_regime_EW,
                      'RISK_PARITY': self._risk_parity_allocation,
                      'VOLATILITY_WEIGHTED': self._volatility_weighted_allocation,
                      'RISK_TARGET': self._risk_targeted_allocation
                      }

        if mode not in self.modes.keys():
            raise ValueError('UNKNOWN MODE "{}"'.format(mode))

        self.caller = None  # portfolio or strategy object calling the model

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_weights(self, context):
        # return self.modes[self.mode](context, elligible, allocations, *args)
        if self.mode.startswith('REGIME') and self.caller.ID != 'algo':
            raise ValueError('ILLEGAL REGIME ALLOCATION : REGIME ALLOCATION MODEL ONLY ALLOWED AT ALGO LEVEL')
        return self.modes[self.mode](context)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _equal_weight_allocation(self, context):
        elligible = self.caller.elligible
        if len(elligible) > 0:
            self.caller.weights = [1. / len(elligible) for i in elligible]
        return self.caller.weights
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _fixed_allocation(self, context):
        # we are going to change these weights, so be careful to keep a copy!
        self.caller.weights = self.caller.allocation_model.weights[:]
        return self.caller.weights

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _proportional_allocation(self, context):
        elligible = self.caller.elligible
        score = self.caller.score
        self.caller.weights = score[elligible] / score[elligible].sum()
        return self.caller.weights
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _risk_parity_allocation(self, context):
        lookback = self.kwargs['lookback']
        prices = context.raw_data['price'][self.caller.elligible][-lookback:]
        ret_log = np.log(1. + prices.pct_change())[1:]
        hist_vol = ret_log.std(ddof=0)

        adj_vol = 1. / hist_vol

        self.caller.weights = adj_vol.div(adj_vol.sum(), axis=0)
        return self.caller.weights
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _volatility_weighted_allocation(self, context):

        elligible = self.caller.elligible

        try:
            hist_vol = context.algo_data['hist_vol'][elligible]
        except:
            raise RuntimeError('No "hist_vol" transform data available')

        adj_vol = 1. / hist_vol

        self.caller.weights = adj_vol.div(adj_vol.sum())
        return self.caller.weights
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _risk_targeted_allocation(self, context):
        lookback = self.kwargs['lookback']
        target_risk = self.kwargs['target_risk']
        shorts = self.kwargs['shorts']
        prices = context.raw_data['price'][self.caller.elligible][-lookback:]
        sigma_mat = self._compute_covariance_matrix(prices)
        mu_vec = self._compute_expected_returns(prices)
        risk_free = context.raw_data['price'][context.risk_free].pct_change()[-lookback:].mean()
        self.caller.weights = self._compute_target_risk_portfolio(mu_vec, sigma_mat,
                                                                  target_risk=target_risk,
                                                                  risk_free=risk_free,
                                                                  shorts=shorts)[0]
        return self.caller.weights
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _min_variance_allocation(self, context):
        lookback = self.kwargs['lookback']
        shorts = self.kwargs['shorts']
        prices = context.raw_data['price'][self.caller.elligible][-lookback:]
        sigma_mat = self._compute_covariance_matrix(prices)
        mu_vec = self._compute_expected_returns(prices)
        self.caller.weights = self._compute_global_min_portfolio(mu_vec=mu_vec,
                                                                 sigma_mat=sigma_mat,
                                                                 shorts=shorts)[0]
        return self.caller.weights
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _max_sharpe_allocation(self, context):
        # calculate security weights for max sharpe portfolio
        lookback = self.kwargs['lookback']
        shorts = self.kwargs['shorts']
        prices = context.raw_data['price'][self.caller.elligible][-lookback:]
        sigma_mat = self._compute_covariance_matrix(prices)
        mu_vec = self._compute_expected_returns(prices)
        risk_free = context.raw_data['price'][context.risk_free].pct_change()[-lookback:].mean()
        self.caller.weights = self._compute_tangency_portfolio(mu_vec=mu_vec,
                                                               sigma_mat=sigma_mat,
                                                               risk_free=risk_free,
                                                               shorts=shorts)[0]
        return self.caller.weights
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _nominal_sharpe_allocation(self, context):
        if isinstance(self.caller, Strategy) == False:
            raise ValueError(
                'Allocation mode {} is only allowed for strategies with fixed weight porfolios'.format(self.mode))
        try:
            portfolio_SRs = [p.metrics.rolling_n_day_SR[-1] for p in self.caller.portfolios]
        except:
            return [0 for p in self.caller.portfolios]

        self.caller.weights = [1. if s == np.max(portfolio_SRs) else 0 for s in portfolio_SRs]
        return self.caller.weights

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _allocation_by_formula(self, context):
        # for Protective Asset Allocation (PAA), strategy assumed to have 2 portfolios
        if self.formula == 'PAA':
            if len(self.caller.elligible) != 2:
                raise ValueError('Protective Asset Allocation (PAA) Srategy has {} Portfolio; must have 2')
            else:
                self.caller.allocations = self._allocate_by_PAA_formula(context)
        return self.caller.allocations

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _allocate_by_PAA_formula(self, context):
        securities = self.caller.portfolios[0].securities
        N = len(securities)
        n = context.rules[self.rule].apply_rule(context)[securities].sum()
        dpf = (N - n) / (N - context.protection_factor * n / 4.)
        # log.debug ('For portfolio {}, n = {}, N = {}, dpf = {}'.format(self.caller.ID, n, N, dpf))
        record('DPF', dpf)
        self.caller.weights = [1. - dpf, dpf]
        return self.caller.weights

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def allocate_by_regime_EW(self, context):

        # log.debug('\nACTIVE : {} \n'.format(self.caller.active))

        self._reset_strategy_and_portfolio_weights(context)

        for strategy in self.caller.strategies:
            strategy.allocations = pd.Series(0, index=strategy.all_assets)

            for pfolio in strategy.portfolios:
                if strategy.ID in self.caller.active:
                    p_weight = 1. / len(strategy.portfolios)
                elif pfolio.ID in self.caller.active:
                    p_weight = 1. / np.sum([1 if p.ID in self.caller.active else 0 for p in strategy.portfolios])
                elif strategy.ID not in self.caller.active and pfolio.ID not in self.caller.active:
                    continue

                pfolio.allocations = pfolio.reallocate(context)
                strategy.allocations = strategy.allocations.add(p_weight * pfolio.allocations, fill_value=0)

        active_strategies = set([s.ID for s in context.strategies
                                 for p in s.portfolios if s.ID in self.caller.active
                                 or p.ID in self.caller.active])
        self.caller.weights = [1. / len(active_strategies) if s.ID in active_strategies else 0 for s in
                               context.strategies]
        context.strategy_weights = self.caller.weights

        return self.caller.weights
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _reset_strategy_and_portfolio_weights(self, context):

        for s_no, s in enumerate(self.caller.strategies):
            self.caller.weights[s_no] = 0
            context.strategy_weights[s_no] = 0
            for p_no, p in enumerate(s.portfolios):
                s.weights[p_no] = 0
        return
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _get_no_of_active_portfolios(self):
        # Note : if strategy in active, all its portfolios are active
        number = 0
        for s in self.caller.strategies:
            if s.ID in self.caller.active:
                # all portfolios are active
                for p in s.portfolios:
                    number += 1
            for p in s.portfolios:
                if p.ID in self.caller.active:
                    number += 1

        return number

    # Portfolio Helper Functions

    # Functions:
    #    1. compute_efficient_portfolio        compute minimum variance portfolio
    #                                            subject to target return
    #    2. compute_global_min_portfolio       compute global minimum variance portfolio
    #    3. compute_tangency_portfolio         compute tangency portfolio
    #    4. compute_efficient_frontier         compute Markowitz bullet
    #    5. compute_portfolio_mu               compute portfolio expected return
    #    6. compute_portfolio_sigma            compute portfolio standard deviation
    #    7. compute_covariance_matrix          compute covariance matrix
    #    8. compute_expected_returns           compute expected returns vector

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_covariance_matrix(self, prices):
        # calculates the cov matrix for the period defined by prices
        returns = np.log(1 + prices.pct_change())[1:]
        excess_returns_matrix = returns - returns.mean()
        return 1. / len(returns) * (excess_returns_matrix.T).dot(excess_returns_matrix)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_expected_returns(self, prices):
        mu_vec = np.log(1 + prices.pct_change(1))[1:].mean()
        return mu_vec

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_portfolio_mu(self, mu_vec, weights_vec):
        if len(mu_vec) != len(weights_vec):
            raise RuntimeError('mu_vec and weights_vec must have same length')
        return mu_vec.T.dot(weights_vec)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_portfolio_sigma(self, sigma_mat, weights_vec):

        if len(sigma_mat) != len(sigma_mat.columns):
            raise RuntimeError('sigma_mat must be square\nlen(sigma_mat) = {}\nlen(sigma_mat.columns) ={}'.
                               format(len(sigma_mat), len(sigma_mat.columns)))
        return np.sqrt(weights_vec.T.dot(sigma_mat).dot(weights_vec))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_efficient_portfolio(self, mu_vec, sigma_mat, target_return, shorts=True):

        # compute minimum variance portfolio subject to target return
        #
        # inputs:
        # mu_vec                  N x 1 DataFrame expected returns
        #                         with index = asset names
        # sigma_mat               N x N DataFrame covariance matrix of returns
        #                         with index = columns = asset names
        # target_return           scalar, target expected return
        # shorts                  logical, allow shorts is TRUE
        #
        # output is portfolio object with the following elements
        #
        # mu_p                   portfolio expected return
        # sig_p                  portfolio standard deviation
        # weights                N x 1 DataFrame vector of portfolio weights
        #                        with index = asset names

        # check for valid inputs
        #

        if len(mu_vec) != len(sigma_mat):
            print("dimensions of mu_vec and sigma_mat do not match")
            raise
        if np.matrix([sigma_mat.ix[i][i] for i in range(len(sigma_mat))]).any() <= 0:
            print('Covariance matrix not positive definite')
            raise

        #
        # compute efficient portfolio
        #

        solvers.options['show_progress'] = False
        P = 2 * matrix(sigma_mat.values)
        q = matrix(0., (len(sigma_mat), 1))
        G = spdiag([-1. for i in range(len(sigma_mat))])
        A = matrix(1., (1, len(sigma_mat)))
        A = matrix([A, matrix(mu_vec.T.values).T], (2, len(sigma_mat)))
        b = matrix([1.0, target_return], (2, 1))

        if shorts == True:
            h = matrix(1., (len(sigma_mat), 1))

        else:
            h = matrix(0., (len(sigma_mat), 1))

        # weights_vec = pd.DataFrame(np.array(solvers.qp(P, q, G, h, A, b)['x']),\
        #                                     sigma_mat.columns)
        weights_vec = pd.Series(list(solvers.qp(P, q, G, h, A, b)['x']), index=sigma_mat.columns)

        #
        # compute portfolio expected returns and variance
        #
        # print ('*** Debug ***\n_compute_efficient_portfolio:\nmu_vec:\n', self.mu_vec, '\nsigma_mat:\n',
        #        self.sigma_mat, '\nweights:\n', self.weights_vec )
        weights_vec.index = mu_vec.index
        mu_p = self._compute_portfolio_mu(mu_vec, weights_vec)
        sigma_p = self._compute_portfolio_sigma(sigma_mat, weights_vec)

        return weights_vec, mu_p, sigma_p
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _compute_global_min_portfolio(self, mu_vec, sigma_mat, shorts=True):

        solvers.options['show_progress'] = False
        P = 2 * matrix(sigma_mat.values)
        q = matrix(0., (len(sigma_mat), 1))
        G = spdiag([-1. for i in range(len(sigma_mat))])
        A = matrix(1., (1, len(sigma_mat)))
        b = matrix(1.0)

        if shorts == True:
            h = matrix(1., (len(sigma_mat), 1))
        else:
            h = matrix(0., (len(sigma_mat), 1))

        # print ('\nP\n\n{}\n\nq\n\n{}\n\nG\n\n{}\n\nh\n\n{}\n\nA\n\n{}\n\nb\n\n{}\n\n'.format(P,q,G,h,A,b))
        # weights_vec = pd.DataFrame(np.array(solvers.qp(P, q, G, h, A, b)['x']),\
        #                                     index=sigma_mat.columns)
        weights_vec = pd.Series(list(solvers.qp(P, q, G, h, A, b)['x']), index=sigma_mat.columns)

        #
        # compute portfolio expected returns and variance
        #
        # print ('*** Debug ***\n_Global Min Portfolio:\nmu_vec:\n', mu_vec, '\nsigma_mat:\n',
        #        sigma_mat, '\nweights:\n', weights_vec)

        mu_p = self._compute_portfolio_mu(mu_vec, weights_vec)
        sigma_p = self._compute_portfolio_sigma(sigma_mat, weights_vec)

        return weights_vec, mu_p, sigma_p

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_efficient_frontier(self, mu_vec, sigma_mat, risk_free=0, points=100, shorts=True):

        efficient_frontier = pd.DataFrame(index=range(points), dtype=object, columns=['mu_p', 'sig_p', 'sr_p', 'wts_p'])

        gmin_wts, gmin_mu, gmin_sigma = self._compute_global_min_portfolio(mu_vec, sigma_mat, shorts=shorts)

        xmax = mu_vec.max()
        if shorts == True:
            xmax = 2 * mu_vec.max()
        for i, mu in enumerate(np.linspace(gmin_mu, xmax, points)):
            w_vec, portfolio_mu, portfolio_sigma = self._compute_efficient_portfolio(mu_vec, sigma_mat, mu,
                                                                                     shorts=shorts)
            efficient_frontier.ix[i]['mu_p'] = w_vec.dot(mu_vec)
            efficient_frontier.ix[i]['sig_p'] = np.sqrt(w_vec.T.dot(sigma_mat.dot(w_vec)))
            efficient_frontier.ix[i]['sr_p'] = (efficient_frontier.ix[i]['mu_p'] - risk_free) / \
                                               efficient_frontier.ix[i]['sig_p']
            efficient_frontier.ix[i]['wts_p'] = w_vec

        return efficient_frontier

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_tangency_portfolio(self, mu_vec, sigma_mat, risk_free=0, shorts=True):

        efficient_frontier = self._compute_efficient_frontier(mu_vec, sigma_mat, risk_free, shorts=shorts)
        index = efficient_frontier.index[efficient_frontier['sr_p'] == efficient_frontier['sr_p'].max()]

        wts = efficient_frontier['wts_p'][index].values[0]
        mu_p = efficient_frontier['mu_p'][index].values[0]
        sigma_p = efficient_frontier['sig_p'][index].values[0]
        sharpe_p = efficient_frontier['sr_p'][index].values[0]

        return wts, mu_p, sigma_p, sharpe_p

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_target_risk_portfolio(self, mu_vec, sigma_mat, target_risk, risk_free=0, shorts=True):

        efficient_frontier = self._compute_efficient_frontier(mu_vec, sigma_mat, risk_free, shorts=shorts)
        if efficient_frontier['sig_p'].max() <= target_risk:
            log.warn('TARGET_RISK {} > EFFICIENT FRONTIER MAXIMUM {}; SETTING IT TO MAXIMUM'.
                     format(target_risk, efficient_frontier['sig_p'].max()))
            index = len(efficient_frontier) - 1
        elif efficient_frontier['sig_p'].min() >= target_risk:
            log.warn('TARGET RISK {} < GLOBAL MINIMUM {}; SETTING IT TO GLOBAL MINIMUM'.
                     format(target_risk, efficient_frontier['sig_p'].max()))
            index = 1
        else:
            index = efficient_frontier.index[efficient_frontier['sig_p'] >= target_risk][0]

        wts = efficient_frontier['wts_p'][index]
        mu_p = efficient_frontier['mu_p'][index]
        sigma_p = efficient_frontier['sig_p'][index]
        sharpe_p = efficient_frontier['sr_p'][index]

        return wts, mu_p, sigma_p, sharpe_p

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class DownsideProtectionModel():
    def __init__(self, context, mode=None, rule=None, formula=None, *args):

        self.mode = mode
        self.rule = rule
        self.formula = formula
        self.args = args

        self.modes = {'BY_RULE': self._by_rule,
                      'RAA': self._apply_RAA,
                      'BY_FORMULA': self._by_formula
                      }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def apply_protection(self, context, allocations, cash_proxy=None, *args):

        # apply downside protection model to cash_proxy, if it fails, set cash_proxy to risk_free

        if context.allow_cash_proxy_replacement:
            if context.raw_data['price'][cash_proxy][-1] < context.algo_data['price'][-43:].mean():
                cash_proxy = context.risk_free
        try:
            new_allocations = self.modes[self.mode](context, allocations, cash_proxy, *args)
            return new_allocations
        except:
            raise ValueError(
                'MODE IMPLEMENTATION ERROR OR DOWNSIDE PROTECTION MODE "{}" DOES NOT EXIST'.format(self.mode))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _by_rule(self, context, allocations, cash_proxy, *args):
        triggers = context.rules[self.rule].apply_rule(context)[allocations.index]
        new_allocations = pd.Series([0 if triggers[a] else allocations[a] for a in allocations.index],
                                    index=allocations.index)
        new_allocations[cash_proxy] = new_allocations[cash_proxy] + (1 - new_allocations.sum())
        return new_allocations

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _apply_RAA(self, context, allocations, cash_proxy, *args):
        excess_returns = context.algo_data['EMOM']

        tmp1 = [0.5 if excess_returns[asset] > 0 else 0. for asset in allocations.index]

        prices = context.algo_data['price']
        MA = context.algo_data['SMMA']

        tmp2 = [0.5 if prices[asset] > MA[asset] else 0. for asset in allocations.index]

        dpf = pd.Series([x + y for x, y in zip(tmp1, tmp2)], index=allocations.index)

        new_allocations = allocations * dpf
        new_allocations[cash_proxy] = new_allocations[cash_proxy] + (1 - np.sum(new_allocations))

        record('BOND EXPOSURE', new_allocations[cash_proxy])

        return new_allocations

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _by_formula(self, context, allocations, cash_proxy, *args):
        if self.formula == 'DPF':
            try:
                new_allocations = self._apply_DPF(context, allocations, cash_proxy, *args)
            except:
                raise ValueError('FORMULA "{}" DOES NOT EXIST OR ERROR CALCULATING FORMULA'.formmat(self.formula))
        return new_allocations

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _apply_DPF(self, context, allocations, cash_proxy, *args):
        securities = args[0][0]
        N = len(securities)
        try:
            triggers = context.rules[self.rule].apply_rule(context)[securities]
        except:
            raise ValueError('UNABLE TO APPLY RULE {}'.format(self.rule))

        num_neg = triggers[triggers == True].count()
        dpf = float(num_neg) / N
        log.info("DOWNSIDE PROTECTION FACTOR = {}".format(dpf))

        new_allocations = (1. - dpf) * allocations
        new_allocations[cash_proxy] += dpf

        return new_allocations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~