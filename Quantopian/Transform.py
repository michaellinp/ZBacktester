import numpy as np
import pandas as pd
import talib
import re

from zipline.api import order, order_target_percent, get_open_orders, get_datetime, record, schedule_function

from talib import BBANDS, DEMA, EMA, HT_TRENDLINE, KAMA, MA, MAMA, MAVP, MIDPOINT, MIDPRICE, SAR, \
    SAREXT, SMA, T3, TEMA, TRIMA, WMA, ADD, DIV, MAX, MAXINDEX, MIN, MININDEX, MINMAX, \
    MINMAXINDEX, MULT, SUB, SUM, BETA, CORREL, LINEARREG, LINEARREG_ANGLE, \
    LINEARREG_INTERCEPT, LINEARREG_SLOPE, STDDEV, TSF, VAR, ADX, ADXR, APO, AROON, \
    AROONOSC, BOP, CCI, CMO, DX, MACD, MACDEXT, MACDFIX, MFI, MINUS_DI, MINUS_DM, MOM, \
    PLUS_DI, PLUS_DM, PPO, ROC, ROCP, ROCR, ROCR100, RSI, STOCH, STOCHF, STOCHRSI, \
    TRIX, ULTOSC, WILLR, ATR, NATR, TRANGE, ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, \
    FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH, AD, ADOSC, OBV, AVGPRICE, MEDPRICE, \
    TYPPRICE, WCLPRICE, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE

TALIB_FUNCTIONS = [BBANDS, DEMA, EMA, HT_TRENDLINE, KAMA, MA, MAMA, MAVP, MIDPOINT, MIDPRICE, SAR, \
                   SAREXT, SMA, T3, TEMA, TRIMA, WMA, ADD, DIV, MAX, MAXINDEX, MIN, MININDEX, MINMAX, \
                   MINMAXINDEX, MULT, SUB, SUM, BETA, CORREL, LINEARREG, LINEARREG_ANGLE, \
                   LINEARREG_INTERCEPT, LINEARREG_SLOPE, STDDEV, TSF, VAR, ADX, ADXR, APO, AROON, \
                   AROONOSC, BOP, CCI, CMO, DX, MACD, MACDEXT, MACDFIX, MFI, MINUS_DI, MINUS_DM, MOM, \
                   PLUS_DI, PLUS_DM, PPO, ROC, ROCP, ROCR, ROCR100, RSI, STOCH, STOCHF, STOCHRSI, TRIX, \
                   ULTOSC, WILLR, ATR, NATR, TRANGE, ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, \
                   LOG10, SIN, SINH, SQRT, TAN, TANH, AD, ADOSC, OBV, AVGPRICE, MEDPRICE, TYPPRICE, \
                   WCLPRICE, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE]

class Transform():
    def __init__(self, context, name='', function='', inputs=[], kwargs={}, outputs=[]):

        self.name = name
        self.function = function
        self.inputs = inputs
        self.kwargs = kwargs
        self.outputs = outputs

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def apply_transform(self, context):

        # transform format [([<data_items>], function, <data_item>, args)]

        context.dp = pd.Panel(context.raw_data)

        if self.function in TALIB_FUNCTIONS:
            return self._apply_talib_function(context)

        elif self.function.__name__.startswith('roll') or self.function.__name__.startswith(
                'expand') or self.function.__name__ == '<lambda>':
            return self._apply_pandas_function(context)

        else:
            return self.function(self, context)

        raise ValueError('UNKNOWN TRANSFORM {}'.format(self.function))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _apply_talib_function(self, context):

        '''
        Routine to apply transform to data provided as a pandas Panel.
        Inputs:
        dp: pandas dataPanel consisting of a DataFrame for each item in ['open', 'high', 'low', 'close', 'volume',
            'price']; each DataFrame has column names = asset names
        inputs : list of dp items to be used as inputs. If empty (=[]), routine will use default input
                        names from the talib function DOC string
        function : talib function name (e.g. RSI, MACD, ADX etc.) - see list of imported functions above
        output_names : list of names for the tranforms DataFrames
        NOTE: names must be unique and there must be a name for each output (some transforms produce more than
                one output e.g MACD produces 3 outputs)
        args : empty list (=[]), in which case default values are obtained from talib function DOC string.
                otherwise, custom parameters may be provided as a list of integers, the parameters matching
                the FULL parameter list, as per the function DOC string

        Outputs:
            pandas DataPanel with new items (DataFrames) appended for each transform output.

        '''

        # parameters = [a for a in self.args]
        parameters = [self.kwargs[key] for key in iter(self.kwargs)]
        if parameters == []:
            parameters = [int(s) for s in re.findall('\d+', self.function.__doc__)]
        data_items = re.findall("(?<=\')\w+", self.function.__doc__)
        if data_items == []:
            inputs = self.inputs
        else:
            inputs = data_items

        for output in self.outputs:
            context.dp[output] = pd.DataFrame(0, index=context.dp.major_axis, columns=context.dp.minor_axis)

        for asset in context.dp.minor_axis:
            data = [context.dp.transpose(2, 1, 0)[asset][i].values for i in inputs]
            args = data + parameters
            transform = self.function(*args)
            if len(transform) == len(self.outputs) or len(transform) > 3:
                pass
            else:
                raise ValueError('** ERROR : must be output_names for each output')

            if len(self.outputs) == 1:
                context.dp[self.outputs[0]][asset] = transform
            else:
                for i, output in enumerate(self.outputs):
                    context.dp[output][asset] = transform[i]

        # for some reason, if you don't do this, then dp.transpose(2,1,0) gives dp[output][asset] as 0 !!
        for name in self.outputs:
            context.dp[name] = context.dp[name]

        return context.dp
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _apply_pandas_function(self, context):

        '''
        Routine to apply pandas function to column(s) of data provided as Pandas DataFrame.
        Allowed functions include all the pandas.rolling_ and pandas.expanding_ functions.
        NOTE: corr and cov are NOT allowed here, but must be implemented as CUSTOM FUNCTIONS
        Inputs:
            dp = Pandas DataPanel with data to be transformed in one (or more) panel items
            NOTE: in the case of CORR or COV, columns contain price data for each stock.
            inputs = name(s) of item(s) containing data to be transformed (DataFrames with columns = asset names)
            function = name of pandas function provided by user (pd.rolling_  or pd.expanding_ )
            args = list of arguments required by function
        Output:
            Pandas DataPanel with appended items containing the transformed data as DataFrames, or,
            as in the case of CORR and COV functions, the item is a DataPanel of correlations/covariances

        '''
        if 'corr' in self.function.__name__ or 'cov' in self.function.__name__:
            raise ValueError('** ERROR: Correlation and Covariance must be implemented as CUSTOM FUNCTIONS')

        for asset in context.dp.minor_axis:
            context.dp[self.outputs[0]] = self.function(context.dp[self.inputs[0]], *self.args)

        return context.dp
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Custom Transforms

    def n_period_momentum(self, context):

        # percentage increase over n periods
        # arg[0] = no of periods
        # arg[1] = period : 'D'|'W'|'M' (day|week||month)

        h = context.dp[self.inputs[0]]

        no_of_periods = self.kwargs['no_of_periods']
        period = self.kwargs['period']

        if period in ['W', 'M']:
            if h.index[-1].date() == get_datetime().date():
                ds = h.resample('M', how='last').pct_change(no_of_periods).iloc[-1]
            else:
                ds = h.resample('M', how='last').pct_change(no_of_periods).iloc[-2]
        else:
            ds = h.pct_change(no_of_periods).iloc[-1]

        df = pd.DataFrame(0, index=h.index, columns=h.columns)
        df.iloc[-1] = ds

        context.dp[self.outputs[0]] = df

        return context.dp
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def simple_mean_monthly_average(self, context):

        h = context.dp[self.inputs[0]]
        lookback = self.kwargs['lookback']
        # ds = h.resample('M').last()[-lookback - 1:-1].mean()
        ds = h.resample('M', how='last')[-lookback - 1:-1].mean()

        df = pd.DataFrame(0, index=h.index, columns=h.columns)
        df.iloc[-1] = ds

        context.dp[self.outputs[0]] = df

        return context.dp

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def momentum(self, context):

        lookback = self.kwargs['lookback']
        ds = context.dp[self.inputs[0]].iloc[-1] / context.dp[self.inputs[0]].iloc[-lookback] - 1

        df = pd.DataFrame(0, index=context.dp.major_axis, columns=context.dp.minor_axis)
        df.iloc[-1] = ds

        context.dp[self.outputs[0]] = df

        return context.dp

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def daily_returns(self, context):

        context.dp[self.outputs[0]] = context.dp['price'].pct_change(1)

        return context.dp

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def excess_momentum(self, context):

        lookback = self.kwargs['lookback']
        ds = context.dp['price'].pct_change(lookback).iloc[-1] - \
             context.dp['price'][context.risk_free].pct_change(lookback).iloc[-1]

        df = pd.DataFrame(0, index=context.dp.major_axis, columns=context.dp.minor_axis)
        df.iloc[-1] = ds

        context.dp[self.outputs[0]] = df

        return context.dp

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def log_returns(self, context):

        try:
            context.dp[self.outputs[0]] = np.log(1. + context.dp['price'].pct_change(1))
        except:
            raise RuntimeError("Inputs must be ['price']")

        return context.dp

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def historic_volatility(self, context):

        lookback = self.kwargs['lookback']
        try:
            ret_log = np.log(1. + context.dp['price'].pct_change())
        except:
            raise RuntimeError("Inputs must be ['price']")

        # this is for pandas < 0.18
        hist_vol = pd.rolling_std(ret_log, lookback)

        # this is for pandas ver > 0.18
        # hist_vol = ret_log.rolling(window=lookback,center=False).std()

        context.dp[self.outputs[0]] = hist_vol * np.sqrt(252 / lookback)

        return context.dp
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def average_excess_return_momentum(self, context):

        '''
        Average Excess Return Momentum

        average_excess_return_momentum is the average of monthly returns in excess of the risk_free rate for multiple
        periods (1,3,6,12 months). In addtion, average momenta < 0 are set to 0.

        '''
        h = context.dp[self.inputs[0]].copy()
        # hm = h.resample('M').last()
        hm = h.resample('M', how='last')
        # hb = h.resample('M').last()[context.risk_free]
        hb = h.resample('M', how='last')[context.risk_free]

        ds = (hm.ix[-1] / hm.ix[-2] - hb.ix[-1] / hb.ix[-2] + hm.ix[-1] / hm.ix[-4]
              - hb.ix[-1] / hb.ix[-4] + hm.ix[-1] / hm.ix[-7] - hb.ix[-1] / hb.ix[-7]
              + hm.ix[-1] / hm.ix[-13] - hb.ix[-1] / hb.ix[-13]) / 22
        ds[ds < 0] = 0
        df = pd.DataFrame(0, index=h.index, columns=h.columns)
        df.iloc[-1] = ds

        context.dp[self.outputs[0]] = df

        return context.dp
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def paa_momentum(self, context):

        ds = context.dp[self.inputs[0]].iloc[-1] / context.dp[self.inputs[1]].iloc[-1] - 1

        df = pd.DataFrame(0, index=context.dp.major_axis, columns=context.dp.minor_axis)
        df.iloc[-1] = ds

        context.dp[self.outputs[0]] = df

        return context.dp

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def slope(self, context):

        lookback = self.kwargs['lookback']
        ds = pd.Series(index=context.dp.minor_axis)
        for asset in context.dp.minor_axis:
            ds[asset] = talib.LINEARREG_SLOPE(context.dp[self.inputs[0]][asset].values
                                              , lookback)[-1]

        df = pd.DataFrame(0, index=context.dp.major_axis, columns=context.dp.minor_axis)
        df.iloc[-1] = ds
        return df