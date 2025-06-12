# main.py

import threading
import os
from tetris_dqn.train import main as train_main
from flask import Flask, render_template

# Lancer l'entraînement dans un thread séparé
def run_training():
    print("🚀 Lancement de l'entraînement DQN...")
    train_main()

threading.Thread(target=run_training, daemon=True).start()

# Lancement du serveur Flask
app = Flask(__name__, template_folder='tetris_dqn/templates')

@app.route('/')
def index():
    return render_template('tetris_live.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
