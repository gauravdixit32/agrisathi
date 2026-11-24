import os

from waitress import serve

from app import app


if __name__ == "__main__":
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "5000"))
    threads = int(os.getenv("APP_THREADS", "4"))
    serve(app, host=host, port=port, threads=threads)
