import redis, os
from rq import Worker, Queue, Connection
from urllib.parse import urlparse

listen = ['high', 'default', 'low']

url = urlparse(os.environ.get("REDIS_URL"))

conn = redis.Redis(host=url.hostname, port=url.port, username=url.username, 
                password=url.password, ssl=True, ssl_cert_reqs=None)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work()