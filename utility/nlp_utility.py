import datetime
class NLP_Utility(object):

    @staticmethod
    def keyword_in_sentence(keyword, sentence, percent):
        true_len = len(keyword) * percent
        hit_count = 0
        for key in keyword:
            if key in sentence:
                hit_count += 1
                if hit_count >= true_len:
                    return True
        return False

    @staticmethod
    def strtime2datetime(time):
        # print time, type(time)
        return datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

