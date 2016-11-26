import numpy as np
import pandas as pd

from Quantopian.Logger import Logger

log = Logger()

class Data():
    def __init__(self, assets):
        self.all_assets = assets
        return

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def update(self, context, data):

        '''

        generates context.raw_data (dictionary of context.max_lookback values)  and context.algo_data
        (dictioanary current values) for  'high', 'open', 'low', 'close', 'volume', 'price' and all
        transforms
        '''

        # dataframe for each of 'high', 'open', 'low', 'close', 'volume', 'price'
        context.raw_data = self.get_raw_data(context, data)

        # log.info ('\n{} GENERATING ALGO_DATA...'.format(get_datetime().date()))

        # add a dataframe for each transform
        context.raw_data = self.generate_frame_for_each_transform(context, data)

        # only need the current value for each security (Series)
        context.algo_data = self.current_algo_data(context, data)

        return context.algo_data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_tradeable_assets(self, data):
        tradeable_assets = [asset for asset in self.all_assets if data.can_trade(asset)]
        if len(self.all_assets) > len(tradeable_assets):
            non_tradeable = [s.symbol for s in self.all_assets if data.can_trade(s) == False]
            log.error('*** FATAL ERROR : MISSING DATA for securities {}'.format(non_tradeable))
            raise ValueError('FATAL ERROR: SEE LOG FOR MISSING DATA')
        return tradeable_assets

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_raw_data(self, context, data):

        context.raw_data = dict()

        tradeable_assets = self.get_tradeable_assets(data)

        for item in ['high', 'open', 'low', 'close', 'volume', 'price']:
            try:
                context.raw_data[item] = data.history(tradeable_assets, item, context.max_lookback, '1d')
            except:
                log.warn('FATAL ERROR: UNABLE TO LOAD HISTORY DATA FOR {}'.format(item))
                # force exit
                raise ValueError(' *** FATAL ERROR : INSUFFICIENT DATA - SEE LOG *** ')

            if np.isnan(context.raw_data[item].values).any():
                # log.warn ('\n WARNING : THERE ARE NaNs IN THE DATA FOR {} \n FILL BACKWARDS.......'
                #           .format([k.symbol for k in context.raw_data[item].keys() if
                #                    np.isnan(context.raw_data[item][k][0])]))
                context.raw_data[item] = context.raw_data[item].bfill()

        return context.raw_data

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def generate_frame_for_each_transform(self, context, data):

        for transform in context.transforms:
            # result = apply_transform(context, transform)
            result = transform.apply_transform(context)
            outputs = transform.outputs
            if type(result) == pd.Panel:
                context.raw_data.update(dict([(o, result[o]) for o in outputs]))
            elif type(result) == pd.DataFrame:
                context.raw_data[outputs[0]] = result
            else:
                log.error('\n INVALID TRANSFORM RESULT\n')

        return context.raw_data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def current_algo_data(self, context, data):

        context.algo_data = dict()
        for k in [key for key in context.raw_data.keys()
                  if type(context.raw_data[key]) == pd.DataFrame]:
            context.algo_data[k] = context.raw_data[k].ix[-1]
            if np.isnan(context.algo_data[k].values).any():
                security = [s.symbol for s in context.raw_data[k].ix[-1].index
                            if np.isnan(context.raw_data[k][s].ix[-1])][0]
                log.warn('*** WARNING: FOR ITEM {} THERE IS A NAN IN THE DATA FOR {}'.format(k, security))
        return context.algo_data