import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import json
import os


# --- Helper functions (same as before) ---
def get_winning_move(move):
    """Returns the move that beats the given move."""
    if move == 'rock':
        return 'paper'
    elif move == 'paper':
        return 'scissors'
    else:
        return 'rock'


def move_to_int(move):
    """Converts a move string to an integer for the ML model."""
    if move == 'rock': return 0
    if move == 'paper': return 1
    if move == 'scissors': return 2


def int_to_move(num):
    """Converts an integer back to a move string."""
    if num == 0: return 'rock'
    if num == 1: return 'paper'
    if num == 2: return 'scissors'


# --- File saving and loading ---
def save_data(human_history, model, stats):
    """Saves the game data to disk."""
    with open('human_history.json', 'w') as f:
        json.dump(human_history, f)
    joblib.dump(model, 'bot_model.joblib')
    with open('game_stats.json', 'w') as f:
        json.dump(stats, f)
    print("Game progress saved.")


def load_data():
    """Loads game data from disk if it exists."""
    human_history = []
    model = None
    stats = {"bot_wins": 0, "human_wins": 0, "ties": 0, "total_rounds": 0}

    if os.path.exists('human_history.json'):
        with open('human_history.json', 'r') as f:
            human_history = json.load(f)
    if os.path.exists('bot_model.joblib'):
        model = joblib.load('bot_model.joblib')
    if os.path.exists('game_stats.json'):
        with open('game_stats.json', 'r') as f:
            stats = json.load(f)

    print("Game progress loaded from files.")
    return human_history, model, stats


# --- Main game logic with persistence ---
def play_game_ml():
    options = ['rock', 'paper', 'scissors']
    history_length = 4
    training_data_min = 100

    # Load previous game data
    human_history, model, stats = load_data()

    print("Let's play Rock, Paper, Scissors using machine learning! Type 'quit' to end the game.")

    while True:
        human_move_str = input("Enter your move (rock, paper, scissors) or 'quit': ").lower()

        if human_move_str == 'quit':
            save_data(human_history, model, stats)
            print("Thanks for playing!")
            break

        if human_move_str not in options:
            print("Invalid move. Please try again.")
            continue

        human_history.append(move_to_int(human_move_str))
        stats["total_rounds"] += 1

        # Train and predict
        if len(human_history) > training_data_min:
            X = []
            y = []
            for i in range(len(human_history) - history_length):
                X.append(human_history[i:i + history_length])
                y.append(human_history[i + history_length])

            model = DecisionTreeClassifier()
            model.fit(X, y)

            last_moves_for_prediction = np.array(human_history[-history_length:]).reshape(1, -1)
            predicted_move_int = model.predict(last_moves_for_prediction)
            predicted_move_str = int_to_move(predicted_move_int[0])  # Use [0] to get the scalar value

            bot_move_str = get_winning_move(predicted_move_str)
            print(f"I predict you will play {predicted_move_str}. I choose {bot_move_str}.")
        else:
            bot_move_str = random.choice(options)
            print(f"I choose {bot_move_str}.")

        # Evaluate round and update stats
        if human_move_str == bot_move_str:
            print("It's a tie!")
            stats["ties"] += 1
        elif get_winning_move(human_move_str) == bot_move_str:
            print("I win!")
            stats["bot_wins"] += 1
        else:
            print("You win!")
            stats["human_wins"] += 1

        # Print stats
        if stats["total_rounds"] > 0:
            bot_win_rate = (stats["bot_wins"] / stats["total_rounds"]) * 100
            print(
                f"Stats: Wins: {stats['bot_wins']}, Losses: {stats['human_wins']}, Ties: {stats['ties']}, Total Rounds: {stats['total_rounds']}")
            print(f"My Win Rate: {bot_win_rate:.2f}%")
            print("-" * 30)


if __name__ == "__main__":
    play_game_ml()
