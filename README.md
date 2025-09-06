# 🤖 Persistent ML Rock, Paper, Scissors Bot

A Python-based Rock, Paper, Scissors bot that learns and adapts to human opponents using a machine learning model. Unlike simple rule-based bots, this bot uses a `DecisionTreeClassifier` to identify and exploit non-random patterns in an opponent's play history to achieve a higher-than-average win rate. The model and game statistics are saved and loaded, allowing the bot to improve over time across multiple game sessions.

## 🌟 Features
*   **Adaptive Learning:** The bot trains a machine learning model (`DecisionTreeClassifier`) on your past moves to predict your next choice.
*   **Persistent Memory:** It saves the human's play history, the trained model, and game statistics to disk (`.json` and `.joblib` files) so it can remember your patterns for future games.
*   **Real-time Statistics:** Displays a live win rate, along with the win/loss/tie count, after each round.
*   **Robust Gameplay:** Includes validation for user input and clear game instructions.

## 🛠️ Technologies
*   **Python 3.x**
*   **scikit-learn:** For the machine learning model (`DecisionTreeClassifier`).
*   **joblib:** For efficient serialization of the trained model.
*   **json:** For saving game statistics and play history.

## 🚀 Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Install dependencies:**
    Make sure you have Python 3.x installed. Then, install the required libraries using pip:
    ```sh
    pip install scikit-learn joblib numpy
    ```

## 🕹️ How to Play

1.  **Run the script** from your terminal:
    ```sh
    python your_bot_file.py
    ```

2.  **Play the game:** The bot will prompt you to enter your move (`rock`, `paper`, or `scissors`).
    ```
    Let's play Rock, Paper, Scissors using machine learning! Type 'quit' to end the game.
    Enter your move (rock, paper, scissors): rock
    ```

3.  **Watch the bot learn:**
    *   For the first few rounds, the bot will play randomly.
    *   After enough training data is collected, it will start making predictions based on your past behavior.

4.  **Quit and Save:** To end the game and save your progress, type `quit`. The next time you run the script, it will load the previously saved data and resume learning from where it left off.

## 💾 Persistence

The bot automatically creates and uses these files to remember game data between sessions:
*   `human_history.json`: Stores a list of your previous moves.
*   `game_stats.json`: Records the bot's win, loss, and tie counts.
*   `bot_model.joblib`: The serialized machine learning model itself.

## 📈 Improving the Bot

*   **Adjusting Parameters:** You can experiment with different values for `history_length` and `training_data_min` to see how it affects the bot's learning speed and accuracy.
*   **Advanced Models:** For more complex pattern recognition, you could replace the `DecisionTreeClassifier` with more advanced models like a `RandomForestClassifier` or a Recurrent Neural Network (RNN).
*   **Extended Gameplay:** The model's predictions are only as good as the data it's trained on. The longer you play, the better the model will become at predicting your moves.

## 🤝 Contribution

Feel free to open issues or submit pull requests if you have suggestions for new features, bug fixes, or improvements.

## 📜 License

This project is licensed under the MIT License.
