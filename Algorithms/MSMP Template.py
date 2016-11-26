import datetime as dt
import pytz

from zipline import TradingAlgorithm, run_algorithm
from zipline.api import get_open_orders, schedule_function
from zipline.api import set_commission, symbol, symbols
from zipline.api import date_rules, time_rules, commission

from Quantopian.algorithm import Algo, AllocationModel, Regime
from Quantopian.Configurator import StrategyParameters, Configurator
from Quantopian.Transform import Transform
from Quantopian.Rule import Rule

from talib import BBANDS, DEMA, EMA, HT_TRENDLINE, KAMA, MA, MAMA, MAVP, MIDPOINT, MIDPRICE, SAR, \
    SAREXT, SMA, T3, TEMA, TRIMA, WMA, ADD, DIV, MAX, MAXINDEX, MIN, MININDEX, MINMAX, \
    MINMAXINDEX, MULT, SUB, SUM, BETA, CORREL, LINEARREG, LINEARREG_ANGLE, \
    LINEARREG_INTERCEPT, LINEARREG_SLOPE, STDDEV, TSF, VAR, ADX, ADXR, APO, AROON, \
    AROONOSC, BOP, CCI, CMO, DX, MACD, MACDEXT, MACDFIX, MFI, MINUS_DI, MINUS_DM, MOM, \
    PLUS_DI, PLUS_DM, PPO, ROC, ROCP, ROCR, ROCR100, RSI, STOCH, STOCHF, STOCHRSI, \
    TRIX, ULTOSC, WILLR, ATR, NATR, TRANGE, ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, \
    FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH, AD, ADOSC, OBV, AVGPRICE, MEDPRICE, \
    TYPPRICE, WCLPRICE, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE

def define_transforms(context):  # Define transforms
    # select transforms required and make sure correct parameters are used
    context.transforms = [
        # Transform(context, name='momentum', function=Transform.n_period_momentum, inputs=['price'],
        #           kwargs={'no_of_periods':3, 'period':'M'}, outputs=['R']),
        Transform(context, name='mom_A', function=ROCP, inputs=['price'], kwargs={'lookback': 43}, outputs=['mom_A']),
        Transform(context, name='mom_B', function=ROCP, inputs=['price'], kwargs={'lookback': 21}, outputs=['mom_B']),
        Transform(context, name='daily_returns', function=Transform.daily_returns, inputs=['price'], kwargs={},
                  outputs=['daily_returns']),
        Transform(context, name='vol_C', function=STDDEV, inputs=['daily_returns'], kwargs={'lookback': 20},
                  outputs=['vol_C']),
        Transform(context, name='slope', function=Transform.slope, inputs=['price'], kwargs={'lookback': 100},
                  outputs=['slope']),
        # Transform(context, name='TMOM', function=Transform.momentum, inputs=['price'], kwargs={'lookback':43}, outputs=['TMOM']),
        # Transform(context, name='MA', function=SMA, inputs=['price'], args=[context.lookback_B], outputs=['MA']),
        Transform(context, name='R', function=Transform.average_excess_return_momentum, inputs=['price'],
                  kwargs={'lookback': 3}, outputs=['R']),
        # Transform(context, name='RMOM', function=Transform.momentum, inputs=['price'], kwargs={'lookback':43}, outputs=['RMOM']),
        # Transform(context, name='TMOM', function=Transform.excess_momentum, inputs=['price'],
        #           kwargs={'lookback':43}, outputs=['TMOM']),
        # Transform(context, name='EMOM', function=Transform.momentum, inputs=['price'], kwargs={'lookback':43},
        #           outputs=['EMOM']),
        # Transform(context, name='volatility', function=STDDEV, inputs=['daily_returns'], kwargs={'lookback':43},
        #           outputs=['volatility']),
        Transform(context, name='smma', function=Transform.simple_mean_monthly_average, inputs=['price'],
                  kwargs={'lookback': 2}, outputs=['smma']),
        # Transform(context, name='mom', function=Transform.paa_momentum, inputs=['price', 'smma'],
        #           kwargs={'lookback':2}, outputs=['mom']),
        # Transform(context, name='smma_12', function=Transform.simple_mean_monthly_average, inputs=['price'],
        #           kwargs={'lookback':12}, outputs=['smma_12'])
    ]

    return context.transforms


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def define_rules(context):  # Define rules
    # select rules required and make sure correct transform names are used
    context.algo_rules = [
        # Rule(context, name='absolute_momentum_rule', rule="'price' < 'smma' "),
        # Rule(context, name='dual_momentum_rule', rule="'TMOM' < 0"),
        Rule(context, name='smma_rule', rule="'price' < 'smma'"),
        # Rule(context, name='complex_rule', rule="'price' < smma or 'TMOM' < 0"),
        # Rule(context, name='momentum_rule', rule="'price' < 'MA'"),
        Rule(context, name='EAA_rule', rule="'R' <= 0"),
        # Rule(context, name='paa_rule', rule="'mom' <= 0"),
        # Rule(context, name='paa_filter', rule="'mom' > 0"),
        # Rule(context, name='momentum_rule1', rule="'price' < 'smma_12'"),
        # Rule(context, name='riskon', rule="'price' > 'smma_12'", apply_to=context.market_proxy),
        # Rule(context, name='riskoff', rule="'price' <= 'smma_12'", apply_to=context.market_proxy),
        Rule(context, name='neutral', rule="'slope' <= 0.1 and 'slope' >= -0.1",
             apply_to=context.market_proxy),
        Rule(context, name='bull', rule="'slope' > 0.1", apply_to=context.market_proxy),
        Rule(context, name='bear', rule="'slope' < -0.1", apply_to=context.market_proxy)
    ]

    return context.algo_rules


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      ]
def initialize(context):
    # these must ALWAYS be present!
    context.transforms = []
    context.algo_rules = []
    context.max_lookback = 63
    context.outstanding = {}  # orders which span multiple days

    context.raw_data = {}

    #############################################################
    # set the following parameters as required

    context.show_positions = True
    # select records to show in algo.show_records()
    context.show_records = True

    # replace cash_proxy with risk_free if cantec.allow_cash_proxY_replacement is True
    # and cash_proxy price is <= average cash_proxy price over last context.cash_proxy_lookback days
    context.allow_cash_proxy_replacement = False
    context.cash_proxy_lookback = 43  # must be <= context.max_lookback

    context.update_metrics = False
    # to calculate Sharpe ratio
    context.calculate_SR = False
    context.SR_lookback = 63  # must be <= context.max_lookback
    context.SD_factor = 0

    # position only changed if percentage change > threshold
    context.threshold = 0.01

    # the following can be changed
    context.market_proxy = symbols('SPY')[0]
    context.risk_free = symbols('SHY')[0]

    set_commission(commission.PerTrade(cost=10.0))
    context.leverage = 1.0
    #################################################################
    # configure strategies

    context.rebalance_interval = 1  # set interval to n = no of periods (default: months)
    # if you want to change default period, change schedule reallocate below

    # Strategy 1

    rs = StrategyParameters(context,
                            name='rs',
                            portfolios=[symbols('MDY', 'EFA'), symbols('TLT', 'IEF', 'AGG')],
                            portfolio_allocation_modes=['EW', 'RISK_TARGET'],
                            portfolio_allocation_kwargs=[{}, {'lookback': 252, 'target_risk': 0.01, 'shorts': False}],
                            security_weights=[None, None],
                            portfolio_allocation_formulas=[None, None],
                            scoring_methods=['RS', None],
                            scoring_factors=[{'+mom_A': 0.65, '+mom_B': 0.35, '-vol_C': 0.},
                                             None],
                            n_tops=[1, 1],
                            protection_modes=['BY_RULE', 'BY_RULE'],
                            protection_rules=['smma_rule', 'smma_rule'],
                            protection_formulas=[None, None],
                            cash_proxies=[symbol('TLT'), symbol('SHY')],
                            strategy_allocation_mode='FIXED',
                            portfolio_weights=[0.6, 0.4],
                            strategy_allocation_formula=None,
                            strategy_allocation_rule=None
                            )
    # Strategy 2

    eaa = StrategyParameters(context,
                             name='eaa',
                             portfolios=[symbols('EEM', 'IEF', 'IEV', 'MDY', 'QQQ', 'TLT', 'XLV')],
                             portfolio_allocation_modes=['PROPORTIONAL'],
                             portfolio_allocation_kwargs=[None],
                             security_weights=[None],
                             portfolio_allocation_formulas=[None],
                             scoring_methods=['EAA'],
                             #   Golden Offensive EAA: wi ~ zi = (1-ci) * ri^2
                             # scoring_factors = [{'R': 2.0, 'C' : 1.0, 'V' : 0.0, 'S' : 1.0, 'eps' : 1e-6}],
                             #   Golden Defensive EAA: wi ~ zi = squareroot( ri * (1-ci) )
                             scoring_factors=[{'R': 1.0, 'C': 1.0, 'V': 0.0, 'S': 0.5, 'eps': 1e-6}],
                             #  Equal Weighted Return: wi ~ zi = ri ^ eps
                             # scoring_factors = [{'R': 1.0, 'C' : 1.0, 'V' : 0.0, 'S' : 1.0, 'eps' : 1e-6}],
                             # Equal Weighted Hedged: wi ~ zi = ( ri * (1-ci) )^eps
                             # scoring_factors = [{'R': 1.0, 'C' : 1.0, 'V' : 0.0, 'S' : 1.0, 'eps' : 1e-6}],
                             #  Scoring Function Test:
                             # scoring_factors = [{'R': 1.0, 'C' : 1.0, 'V' : 1.0, 'S' : 1.0, 'eps' : 0.0}],
                             # scoring_factors = [{'R': 1.0, 'C' : 1.0, 'V' : 1.0, 'S' : 0.0, 'eps' :  1e-6}],
                             n_tops=[2],
                             protection_modes=['BY_FORMULA'],
                             protection_rules=['EAA_rule'],
                             protection_formulas=['DPF'],
                             cash_proxies=[symbol('TLT')],
                             strategy_allocation_mode='FIXED',
                             portfolio_weights=[1.],
                             strategy_allocation_formula=None,
                             strategy_allocation_rule=None
                             )

    Configurator(context, define_transforms, define_rules, strategies=[rs, eaa])

    ############################
    # configure algorithm

    algo = Algo(context,
                strategies=[rs.strategy, eaa.strategy],
                allocation_model=AllocationModel(context, mode='REGIME_EW', weights=None, formula=None),
                regime=Regime(
                    transitions={'0': ('neutral', ['rs']),
                                 '1': ('bull', ['rs_p1']),
                                 '-1': ('bear', ['eaa'])
                                 }
                ),
                )
    ###########################################################################################################
    # generate algo data every day at close
    schedule_function(algo.update_data, date_rules.every_day(), time_rules.market_close())

    # daily functions to handle GTC orders
    schedule_function(algo.check_for_unfilled_orders, date_rules.every_day(), time_rules.market_close())
    schedule_function(algo.fill_outstanding_orders, date_rules.every_day(), time_rules.market_open())

    if context.update_metrics:
        # calculate metrics every day
        schedule_function(algo.update_metrics, date_rules.every_day(), time_rules.market_close())

    if context.show_positions:
        schedule_function(algo.show_positions, date_rules.month_start(days_offset=0), time_rules.market_open())

    if context.show_records:
        # show records every day
        # edit the show_records function to include records required
        schedule_function(algo.show_records, date_rules.every_day(), time_rules.market_close())

    schedule_function(algo.rebalance, date_rules.month_end(days_offset=2), time_rules.market_open())
###########################################################################################################
#MAIN ROUTINE

capital_base = 10000
start = dt.datetime(2008, 12, 1, 0, 0, 0, 0, pytz.utc)
end = dt.datetime(2016, 11, 1, 0, 0, 0, 0, pytz.utc)

result = run_algorithm(start=start, end=end, initialize=initialize, \
                       capital_base=capital_base, \
                       bundle='etf_bundle')

result.portfolio_value.plot(figsize=(15,10), grid=True)
result.to_pickle('E:\\NOTEBOOKS\\Quantopian\\Strategies\\result.pkl')