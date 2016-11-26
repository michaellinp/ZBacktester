# dummy logger
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
