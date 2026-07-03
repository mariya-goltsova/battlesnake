"""Standalone Flask entry point for the src_rl bot (the root backend.py is untouched).

Run from the repo root:  PORT=8100 python -m src_rl.backend_rl
Endpoints mirror the official Battlesnake API: GET /, POST /start /move /end.
"""

import logging
import os

from flask import Flask, request

from src_rl.policy import choose_move, get_info

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)


@app.get("/")
def index():
    return get_info()


@app.post("/start")
def start():
    return "ok"


@app.post("/move")
def move():
    return {"move": choose_move(request.get_json())}


@app.post("/end")
def end():
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8100")))
