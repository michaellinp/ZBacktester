import numpy as np
from zipline.api import symbols

from Quantopian.algorithm import AllocationModel, DownsideProtectionModel, Strategy, Portfolio, ScoringModel
from Quantopian.Transform import Transform

VALID_PORTFOLIO_ALLOCATION_MODES = ['EW', 'FIXED', 'PROPORTIONAL', 'MIN_VARIANCE', 'MAX_SHARPE', 'NOMINAL_SHARPE',
                                    'BY_FORMULA', 'RISK_PARITY', 'VOLATILITY_WEIGHTED', 'RISK_TARGET']
VALID_PORTFOLIO_ALLOCATION_FORMULAS = [None]
VALID_SCORING_METHODS = [None, 'RS', 'EAA']
VALID_PROTECTION_MODES = [None, 'BY_RULE', 'RAA', 'BY_FORMULA']
VALID_PROTECTION_FORMULAS = [None, 'DPF']
VALID_STRATEGY_ALLOCATION_MODES = ['EW', 'FIXED', 'NOMINAL_SHARPE', 'BY_FORMULA']
VALID_STRATEGY_ALLOCATION_FORMULAS = [None, 'PAA']
NONE_NOT_ALLOWED = ['portfolios', 'portfolio_allocation_modes', 'cash_proxies', 'strategy_allocation_mode']

class StrategyParameters ():

    '''
    StrategyParameters hold the parameters for each strategy for a single or multistrategy algoritm

    calling:

    strategy = StrategyParameters(context, portfolios, portfolio_allocation_modes, security_weights,
                         portfolio_allocation_formulas,
                         scoring_methods, scoring_factors, n_tops,
                         protection_modes, protection_rules, protection_formulas,
                         cash_proxies, strategy_allocation_mode, portfolio_weights=None,
                         strategy_allocation_formula, strategy_allocation_rule)

    where:

    - N = no of portfolios >=1
    - portfolios = list of N lists of valid assets eg. [symbols('SPY','MDY'), symbols('IHF'), symbols('EEM')]
    - portfolio_allocation_modes = list of N valid PORTFOLIO_ALLOCATION_MODES
    - security_weights = list of N lists of security weights, one for each security, sum = 1 (only required if
        portfolio_allocation_mode is 'FIXED')
    - portfolio_allocation_formulas = list of N valid formulas
    - portfolio_allocation_kwargs = list of keyword arguments
    - scoring_methods = list of N VALID_SCORING_METHODS
    - scoring_factors = list of N dictionaries of the form {factor1:weight1, ......factor_n:weight_n}
    - n_tops = list of N integers 0 < n <= no of securities in portfolio
    - protection_modes = list of N VALID_PROTECTION_MODES
    - protection_rules = list of N valid protection rules
    - protection_formulas = list of N valid protection formulas
    - cash_proxies = list of N valid assets eg. symbols('TLT', 'TLT', 'SHY')
    - strategy_allocation_mode = valid strategy allocation mode
    - portfolio_weights = if strategy_allocation_mode = 'FIXED', list of portfolio weights for each portfolio
    - strategy_allocation_formula = valid strategy allocation formula
    - strategy_allocation_rule = valid strategy allocation rule

    '''

    def __init__(self, context, name, portfolios=[], portfolio_allocation_modes=[], security_weights=[],
                 portfolio_allocation_formulas=None, portfolio_allocation_kwargs=None,
                 scoring_methods=[], scoring_factors=[], n_tops=[],
                 protection_modes=[], protection_rules=[], protection_formulas=[],
                 cash_proxies=[],
                 strategy_allocation_mode='FIXED', portfolio_weights=[], strategy_allocation_formula=None,
                 strategy_allocation_rule=None):
        self.name = name
        self.portfolios = portfolios
        self.portfolio_allocation_modes = portfolio_allocation_modes
        self.portfolio_allocation_kwargs = portfolio_allocation_kwargs
        self.security_weights = security_weights
        self.portfolio_allocation_formulas = portfolio_allocation_formulas
        self.scoring_methods = scoring_methods
        self.scoring_factors = scoring_factors
        self.n_tops = n_tops
        self.protection_modes = protection_modes
        self.protection_rules = protection_rules
        self.protection_formulas = protection_formulas
        self.cash_proxies = cash_proxies
        self.strategy_allocation_mode = strategy_allocation_mode
        self.portfolio_weights = portfolio_weights
        self.strategy_allocation_formula = strategy_allocation_formula
        self.strategy_allocation_rule = strategy_allocation_rule

class Configurator():

    '''


    The Configurator uses the Strategy Parameters set up by the StrategyParameters Class and dictionary of global
    parameters to create all the objects required for the algorithm.

    '''

    def __init__(self, context, define_transforms, define_rules, strategies):
        self.strategies = strategies
        # self.global_parameters = global_parameters
        # self._set_global_parameters (context)
        context.tranforms = define_transforms(context)
        if Transform.average_excess_return_momentum in [t.function for t in context.transforms]:
            context.max_lookback = max(274, context.max_lookback)
        context.algo_rules = define_rules(context)
        self._configure_algo_strategies(context)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _configure_algo_strategies(self, context):
        for s in self.strategies:
            self._check_valid_parameters(context, s)
            self._configure_strategy(context, s)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _check_valid_parameters(self, context, strategy):
        N = len(strategy.portfolios)
        s = strategy
        self._check_valid_parameter(context, s, strategy.portfolios, 'portfolios', list, N, list, ''),
        self._check_valid_parameter(context, s, strategy.portfolio_allocation_modes, 'portfolio_allocation_modes',
                                    list, N, str, VALID_PORTFOLIO_ALLOCATION_MODES),
        self._check_valid_parameter(context, s, strategy.security_weights, 'security_weights', list, N, list, ''),
        self._check_valid_parameter(context, s, strategy.portfolio_allocation_formulas, 'portfolio_allocation_formulas',
                                    list,
                                    N, str, VALID_PORTFOLIO_ALLOCATION_FORMULAS),
        self._check_valid_parameter(context, s, strategy.scoring_methods, 'scoring_methods', list, N,
                                    str, VALID_SCORING_METHODS),
        self._check_valid_parameter(context, s, strategy.scoring_factors, 'scoring_factors', list, N, dict, ''),
        self._check_valid_parameter(context, s, strategy.n_tops, 'n_tops', list, N, int, '')
        self._check_valid_parameter(context, s, strategy.protection_modes, 'protection_modes', list, N,
                                    str, VALID_PROTECTION_MODES),
        self._check_valid_parameter(context, s, strategy.protection_rules, 'protection_rules', list, N, str, ''),
        self._check_valid_parameter(context, s, strategy.protection_formulas, 'protection_formulas', list, N,
                                    str, VALID_PROTECTION_FORMULAS),
        self._check_valid_parameter(context, s, strategy.cash_proxies, 'cash_proxies', list, N, type(symbols('SPY')[0]),
                                    ''),
        self._check_valid_parameter(context, s, strategy.strategy_allocation_mode, 'strategy_allocation_mode', str,
                                    1, str, VALID_STRATEGY_ALLOCATION_MODES)
        self._check_valid_parameter(context, s, strategy.portfolio_weights, 'portfolio_weights', list, N, float, ''),
        self._check_valid_parameter(context, s, strategy.strategy_allocation_formula, 'strategy_allocation_formula',
                                    str,
                                    1, str, VALID_STRATEGY_ALLOCATION_FORMULAS)
        self._check_valid_parameter(context, s, strategy.strategy_allocation_rule, 'strategy_allocation_rule', str,
                                    1, str, '')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _check_valid_parameter(self, context, s, param, name, param_type, param_length, item_type, valid_params):

        if name in ['strategy_allocation_mode', 'portfolio_weights', 'strategy_allocation_formula',
                    'strategy_parameters', 'strategy_allocation_rule']:
            self._check_strategy_parameters(context, s, param, name, param_type, param_length, item_type, valid_params)
        else:
            self._check_param_type(name, param, param_type)
            if len(param) != param_length:
                raise RuntimeError('Parameter {} must be of length {} not {}'.format(name, param_length, len(param)))
            for n in range(param_length):
                if param[n] == None and name in NONE_NOT_ALLOWED:
                    raise RuntimeError('"None" not allowed for parameter {}'.format(name))
                elif param[n] == None:
                    continue
                if valid_params != "":
                    if param[n] not in valid_params:
                        raise RuntimeError('Invalid {} {}'.format(name, param[n]))
                if not isinstance(param[n], item_type):
                    raise RuntimeError('Items of {} must be of type {} not {}'.format(name, item_type, type(param[n])))
                if name == 'portfolios':
                    self._check_valid_portfolio(param[n])
                if name == 'scoring_factors' and s.protection_modes == 'RS':
                    self._check_valid_scoring_factors(name, param[n])
                if name == 'n_tops' and s.portfolio_allocation_modes[n] == 'FIXED':
                    if param[n] != len(s.security_weights[n]):
                        raise RuntimeError(
                            'For portfolio_allocation_mode = "FIXED", n_tops must equal no of security weights')
                if name.endswith('_weights') and np.sum(param[n]) != 1.:
                    raise RuntimeError('Sum of {} must equal 1, not {}'.format(name, np.sum(param)))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_strategy_parameters(self, context, s, param, name, param_type, param_length, item_type, valid_params):
        if name == 'strategy_allocation_mode':
            if param not in valid_params:
                raise RuntimeError('Invalid strategy_allocation_mode {}'.format(param))
        elif name == 'portfolio_weights' and s.strategy_allocation_mode == 'FIXED':
            if np.sum(param) != 1.:
                raise RuntimeError('portfolio_weights must be a list of floating point numbers with sum = 1')
        elif name == 'strategy_allocation_formula':
            if param not in valid_params:
                raise RuntimeError('Invalid strategy_allocation_formula {}'.format(param))
        if name == 'strategy_allocation_rule' and s.strategy_allocation_rule != None:
            valid_rules = [rule.name for rule in context.algo_rules]
            if s.strategy_allocation_rule not in valid_rules:
                raise RuntimeError(
                    'Strategy rule {} not found. Check rule definitions'.format(s.strategy_allocation_rule))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_param_type(self, name, param, param_type):
        if not isinstance(param, param_type):
            raise RuntimeError('Parameter {} must be of type {} not {}'.format(name, param_type, type(param)))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_valid_scoring_factors(self, name, factors):
        sum_of_weights = 0.

        for key in factors.keys():
            if not key[0] in ['+', '-']:
                raise RuntimeError('First character of scoring factor {}, must be "+" or "-"'.format(key))
            sum_of_weights += factors[key]
        if sum_of_weights != 1.:
            raise RuntimeError('Sum of {} weights must equal 1, not {}'.format(name, sum_of_weights))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_valid_portfolio(self, pfolio):
        if len(pfolio) < 1:
            raise RuntimeError('Portfolio must have at least one item')
        for n in range(len(pfolio)):
            if not isinstance(pfolio[n], type(symbols('SPY')[0])):
                raise RuntimeError('portfolio item {} must be of type '.format(type(symbols('SPY')[0])))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _configure_strategy(self, context, s):

        portfolios = []

        for n in range(len(s.portfolios)):
            if s.scoring_factors[n] == None:
                scoring_model = None
            else:
                scoring_model = ScoringModel(context,
                                             method=s.scoring_methods[n],
                                             factors=s.scoring_factors[n],
                                             n_top=s.n_tops[n])

            if s.protection_modes[n] == None:
                downside_protection_model = None
            else:
                downside_protection_model = DownsideProtectionModel(context,
                                                                    mode=s.protection_modes[n],
                                                                    rule=s.protection_rules[n],
                                                                    formula=s.protection_formulas[n])

            portfolios = portfolios + \
                         [Portfolio(context,
                                    ID=s.name + '_p' + str(n + 1),
                                    securities=s.portfolios[n],
                                    allocation_model=AllocationModel(context,
                                                                     mode=s.portfolio_allocation_modes[n],
                                                                     weights=s.security_weights[n],
                                                                     formula=s.portfolio_allocation_formulas[n],
                                                                     kwargs=s.portfolio_allocation_kwargs[n]
                                                                     ),
                                    scoring_model=scoring_model,
                                    downside_protection_model=downside_protection_model,
                                    cash_proxy=s.cash_proxies[n]
                                    )]

        s.strategy = Strategy(context,
                              ID=s.name,
                              allocation_model=AllocationModel(context,
                                                               mode=s.strategy_allocation_mode,
                                                               weights=s.portfolio_weights,
                                                               formula=s.strategy_allocation_formula,
                                                               rule=s.strategy_allocation_rule),
                              portfolios=portfolios
                              )