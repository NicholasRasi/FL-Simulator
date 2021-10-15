def auto_str(cls):
    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )
    cls.__str__ = __str__
    return cls


def print_queue_state(num_jobs: int):
    str = ''
    for x in range(num_jobs):
        str = str + u"\u2588 "
    print("Queue state: " + str)
