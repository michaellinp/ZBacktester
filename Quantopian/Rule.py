class Rule():

    functions = {'EQ': lambda x, y: x == y,
                 'LT': lambda x, y: x < y,
                 'GT': lambda x, y: x > y,
                 'LE': lambda x, y: x <= y,
                 'GE': lambda x, y: x >= y,
                 'NE': lambda x, y: x != y,
                 'AND': lambda x, y: x & y,
                 'OR': lambda x, y: x | y
                 }

    def __init__(self, context, name='', rule='', apply_to='all'):

        self.name = name
        # remove spaces
        self.rule = rule.replace(' ', '')
        self.temp = ''
        self.apply_to = apply_to

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def apply_rule(self, context):

        ''' routine to evaluate a rule consisting of a string formatted as 'conditional [AND|OR conditional]'
            where conditionals are logical expressions, pandas series of logical expressions
            or pandas dataframes of logical expressions. Returns True or False,
            pandas series of True/False or pandas dataframe of True/False respectively.
        '''

        if self.rule == 'always_true':
            return True

        self.temp = self._replace_operators(self.rule)
        # get the first condition of the rule and evaluate it
        condition, result, cjoin = self._get_next_conditional(context)

        # log.debug ('result = {}'.format(result))

        while cjoin != None:
            # get the rest of the rule
            self.temp = self.temp[len(condition) + len(cjoin):]
            # get the next conditional of the rule and evaluate it
            func = Rule.functions[cjoin]
            condition, tmp_result, cjoin = self._get_next_conditional(context)

            result = func(result, tmp_result)

            # log.debug ('intermediate result = {}'.format(result))

        # log.debug ('final result = {}'.format(result))
        return result

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_next_conditional(self, context):
        condition, cjoin = self._get_conditional(self.temp)
        result = self._evaluate_condition(context, condition)
        if self.apply_to != 'all':
            result = result[self.apply_to]
        return condition, result, cjoin
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _replace_operators(self, s):

        ''' to make it easy to find operators in the rule s, replace ['=', '>', '<', '>=', '<=', '!=', 'and', 'or']
            with ['EQ', 'GT', 'LT', 'GE', 'LE', 'NE', 'AND', 'OR'] respectively
        '''

        s1 = s.replace('and', 'AND').replace('or', 'OR').replace('!=', 'NE').replace('<=', 'LE').replace('>=', 'GE')
        s1 = s1.replace('=', 'EQ').replace('<', 'LT').replace('>', 'GT')
        return s1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_conditional(self, s):

        ''' routine to find first ocurrence of "AND" or "OR" in rule s. Returns
        conditional to the left of AND/OR and either 'AND', 'OR' or None '''

        pos_AND = [s.find('AND') if s.find('AND') != -1 else len(s)][0]
        pos_OR = [s.find('OR') if s.find('OR') != -1 else len(s)][0]
        condition, cjoin = [(s.split('AND')[0], 'AND') if pos_AND < pos_OR else (s.split('OR')[0], 'OR')][0]
        if pos_AND == len(s) and pos_OR == len(s):
            cjoin = None
        return condition, cjoin

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_operator(self, condition):

        '''routine to extract the operator and its position from the conditional expression
        '''
        for o in ['EQ', 'GT', 'LT', 'GE', 'LE', 'NE', 'AND', 'OR']:
            if condition.find(o) > 0:
                return o, condition.find(o)
        raise ('UNKNOWN OPERATOR')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_operand_value(self, context, operand):
        if operand.startswith('('):
            tuple_0 = operand[1:operand.find(',')].strip("'").strip('"')
            tuple_1 = operand[operand.find(',') + 1:-1]
            return context.algo_data[tuple_0][tuple_1]
        if operand[0].isdigit() or operand.startswith('.') or operand.startswith('-'):
            return float(operand)
        elif isinstance(operand, str):
            return context.algo_data[operand.strip("'").strip('"')]
        else:
            op = context.algo_data[operand[0]]
            if operand[1] != None:
                op = context.algo_data[operand[0].strip("'").strip('"')][operand[1]]
            return op

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _evaluate_condition(self, context, condition):
        operator, position = self._get_operator(condition)
        x = self._get_operand_value(context, condition[:position])
        y = self._get_operand_value(context, condition[position + 2:])
        # log.debug ('x = {}, y = {}, operator = {}'.format(x, y, operator))
        func = Rule.functions[operator]

        return func(x, y)