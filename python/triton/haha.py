times = {}
counter = {}

compile_kwargs = {}


def record_timer(key, duration):
    times[key] = times.get(key, 0.) + duration
    counter[key] = counter.get(key, 0) + 1
