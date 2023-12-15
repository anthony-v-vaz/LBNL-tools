class Counter(object):
    def __init__(self):
        self.value = 0
   
    def increment(self):
        self.value += 1
        return self.value

from dask.distributed import Client

client = Client()

# Create 10 Actors, and call increment() once on each of them
counters = [client.submit(Counter, actor=True).result() for _ in range(10)]
results = [c.increment().result() for c in counters]
print(results)

# Call increment() 5 times on the first Actor in the list
results = [counters[0].increment().result() for _ in range(5)]
print(results)


