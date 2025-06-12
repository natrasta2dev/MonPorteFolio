import sys
import os

# Définir le chemin du répertoire tetris_dqn
tetris_path = os.path.join(os.path.dirname(__file__), 'tetris_dqn')

# Ajouter le répertoire tetris_dqn au path Python
sys.path.append(tetris_path)

# Importer et exécuter le train.py du sous-dossier
from train import main

if __name__ == '__main__':
    main()
