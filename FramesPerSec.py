from datetime import datetime
import time

class FramesPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._start_time = None
        self._num_frames = 0

    def current_milli_time(self):
        return int(round(time.time() * 1000))
    
    def start(self):
        self._start_time = self.current_milli_time()#datetime.now()
        return self

    def increment(self):
        self._num_frames += 1

    def fps(self):
        elapsed_time_millis = float(self.current_milli_time() - self._start_time)
        if(elapsed_time_millis == 0):
            elapsed_time_millis = 1
        return float(self._num_frames / float(elapsed_time_millis / 1000.0))
