import redis, os
from rq import Worker, Queue, Connection
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(r'C:\Users\12482\Desktop\blockchain\dApp\app\.env'))

listen = ['high', 'default', 'low']

redis_url = os.getenv('REDIS_URL')

conn = redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work()