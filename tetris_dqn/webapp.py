# tetris_dqn/webapp.py

from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('tetris_live.html')  # ton interface front

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Port utilisé par Render
    app.run(host='0.0.0.0', port=port)
