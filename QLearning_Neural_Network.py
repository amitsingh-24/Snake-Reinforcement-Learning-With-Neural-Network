

import random
from collections import deque
from Snake import SnakeGame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the neural network model
def create_model(input_dim, output_dim):
    model = Sequential([
        Dense(24, input_dim=input_dim, activation='relu'),
        Dense(24, activation='relu'),
        Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Modify the evaluateScore function to use the neural network
def evaluateScore(model, boardDim, numRuns, displayGame=False):
    scores = []
    for _ in range(numRuns):
        game = SnakeGame(boardDim, boardDim)
        state = game.calcStateNum()
        score = 0
        gameOver = False
        while not gameOver:
            state_vector = tf.one_hot(state, depth=256)  # One-hot encode the state
            state_input = tf.expand_dims(state_vector, 0)  # Reshape to match the input shape expected by the model
            action_probs = model(state_input, training=False)
            action = np.argmax(action_probs[0].numpy())
            state, _, gameOver, score = game.makeMove(action)
        scores.append(score)
    return np.average(scores), scores

# %%
boardDim = 16  # size of the board
numStates = 2**8
numActions = 4  # 4 directions that the snake can move
gamma = 0.8  # discount rate
epsilon = 0.2  # exploration rate in training games
numEpisodes = 1001  # number of games to train for
batch_size = 32  # batch size for training
memory = deque(maxlen=2000)  # replay buffer

# Create the neural network model
model = create_model(numStates, numActions)

# Training loop
bestLength = 0
print("Training for", numEpisodes, "games...")
for episode in range(numEpisodes):
    game = SnakeGame(boardDim, boardDim)
    state = game.calcStateNum()
    gameOver = False
    score = 0
    while not gameOver:
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)
        else:
            state_vector = tf.one_hot(state, depth=256)
            state_input = tf.expand_dims(state_vector, 0)
            action_probs = model(state_input, training=False)
            action = np.argmax(action_probs[0].numpy())
        new_state, reward, gameOver, score = game.makeMove(action)
        
        # Store experience in replay buffer
        memory.append((state, action, reward, new_state, gameOver))
        
        state = new_state

        # Train the model with a batch of experiences
        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            for state, action, reward, new_state, done in batch:
                target = reward
                if not done:
                    target = reward + gamma * np.max(model.predict(tf.expand_dims(tf.one_hot(new_state, depth=256), 0)))
                target_f = model.predict(tf.expand_dims(tf.one_hot(state, depth=256), 0))
                target_f[0][action] = target
                model.fit(tf.expand_dims(tf.one_hot(state, depth=256), 0), target_f, epochs=1, verbose=0)
    
    if episode % 100 == 0:
        averageLength, lengths = evaluateScore(model, boardDim, 25)
        if averageLength > bestLength:
            bestLength = averageLength
        print("Episode", episode, "Average snake length without exploration:", averageLength)

# The rest of the code remains the same for animation and visualization

#%%
#Animate games at different episodes
print("Generating data for animation...")
plotEpisodes = [0, 200, 300, 400, 500, 600, 700, 800, 900]
# plotEpisodes = [0, 200, 400, 600, 800, 1000, 2500, 5000, 10000]
fig, axes = plt.subplots(3, 3, figsize=(9,9))

axList = []
ims = []
dataArrays = []
scores = []
labels = []

for i, row in enumerate(axes):
    for j, ax in enumerate(row):
        ax.set_title("Episode " + str(plotEpisodes[i*len(row) + j]))
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        axList.append(ax)
        ims.append(ax.imshow(np.zeros([boardDim, boardDim]), vmin=-1, vmax=1, cmap='RdGy'))
        labels.append(ax.text(0,15, "Length: 0", bbox={'facecolor':'w', 'alpha':0.75, 'pad':1, 'edgecolor':'white'}))
        dataArrays.append(list())
        scores.append(list())
        
stopAnimation = False
maxFrames = 1000
cutoff = 100
numGames = 10
for k in range(numGames):
    games = []
    states = []
    gameOvers = []
    moveCounters = []
    oldScores = []
    for l in range(len(plotEpisodes)):
        game = SnakeGame(boardDim, boardDim)
        games.append(game)
        states.append(game.calcStateNum())
        gameOvers.append(False)
        moveCounters.append(0)
        oldScores.append(0)
    for j in range(maxFrames):
        for i in range(len(plotEpisodes)):
            possibleQs = Qs[plotEpisodes[i], :, :][states[i], :]
            action = np.argmax(possibleQs)
            states[i], reward, gameOver, score = games[i].makeMove(action)
            if gameOver:
                gameOvers[i] = True
            dataArrays[i].append(games[i].plottableBoard())
            scores[i].append(score)
            if score == oldScores[i]:
                moveCounters[i] += 1
            else:
                oldScores[i] = score
                moveCounters[i] = 0
            if moveCounters[i] >= cutoff:
                # stuck going back and forth
                gameOvers[i] = True
        if not any(gameOver == False for gameOver in gameOvers):
            print("Game", k, "finished, total moves:", len(dataArrays[0]))
            break

def animate(frameNum):
    for i, im in enumerate(ims):
        labels[i].set_text("Length: " + str(scores[i][frameNum]))
        ims[i].set_data(dataArrays[i][frameNum])
    return ims+labels
print("Animating snakes at different episodes...")

numFrames = len(dataArrays[0])
ani = animation.FuncAnimation(fig, func=animate, frames=numFrames,blit=True, interval=75, repeat=False, )
plt.show(block=False)
#%%
##uncomment below if you want to output to a video file
print("Saving to file")
ani.save('AnimatedGames.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
print("Done")













# import os
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import tensorflow as tf
# from tensorflow.keras import layers
# from Snake import SnakeGame

# print(animation.writers.list())


# # Define the Neural Network Model for Q-Learning
# def create_q_model(num_states, num_actions):
#     inputs = layers.Input(shape=(num_states,))
#     layer1 = layers.Dense(128, activation="relu")(inputs)
#     layer2 = layers.Dense(128, activation="relu")(layer1)
#     action = layers.Dense(num_actions, activation="linear")(layer2)
#     return tf.keras.Model(inputs=inputs, outputs=action)

# # # Modify the evaluateScore function to use the neural network
# # def evaluateScore(model, boardDim, numRuns, displayGame=False):
# #     print("Starting evaluateScore")
# #     scores = []
# #     for _ in range(numRuns):
# #         game = SnakeGame(boardDim, boardDim)
# #         state = game.calcStateNum()
# #         score = 0
# #         gameOver = False
# #         while not gameOver:
# #             # Assuming state is a scalar or not in an array form, wrap it in an array first
# #             state_vector = tf.one_hot(state, depth=256)  # One-hot encode the state
# #             state_input = tf.expand_dims(state_vector, 0)  # Reshape to match the input shape expected by the model

# #             action_probs = model(state_input, training=False)

# #             action = np.argmax(action_probs[0].numpy())
# #             state, _, gameOver, score = game.makeMove(action)
# #         scores.append(score)
# #     print("Finished evaluateScore")
# #     return np.average(scores), scores

# # Parameters
# boardDim = 16
# numStates = 2**8
# numActions = 4
# gamma = 0.8
# epsilon = 0.2
# numEpisodes = 1001

# # Initialize the Neural Network Model
# model = create_q_model(numStates, numActions)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
# loss_function = tf.keras.losses.MeanSquaredError()


# # Training Loop
# for episode in range(numEpisodes):
#     game = SnakeGame(boardDim, boardDim)
#     state = game.calcStateNum()
#     gameOver = False
#     while not gameOver:
#         if random.uniform(0, 1) < epsilon:
#             action = random.randint(0, 3)
#         else:
#             # Convert state to a tensor with a batch dimension
#             state_vector = tf.one_hot(state, depth=256)

#             # Ensure state_input has the correct shape (1, 256)
#             state_input = tf.expand_dims(state_vector, 0)

#             # Now, state_input can be passed to the model
#             action_probs = model(state_input, training=False)
#             action = np.argmax(action_probs[0].numpy())
#         new_state, reward, gameOver, _ = game.makeMove(action)
#         # Assuming new_state is calculated similarly to state and is a scalar or not in an array form
#         new_state_vector = tf.one_hot(new_state, depth=256)

#         # Ensure new_state_input has the correct shape (1, 256) for the model
#         new_state_input = tf.expand_dims(new_state_vector, 0)

#         # Now, new_state_input can be passed to the model
#         future_rewards = model.predict(new_state_input)
#         updated_q = reward + gamma * np.max(future_rewards)
#         with tf.GradientTape() as tape:
#             state_vector = tf.one_hot(state, depth=256)

#             # Ensure state_vector has the correct shape (1, 256) for the model
#             state_input = tf.expand_dims(state_vector, 0)

#             # Now, state_input can be passed to the model
#             q_values = model(state_input, training=True)
#             q_value = q_values[0, action]
#             updated_q_tensor = tf.expand_dims(updated_q, axis=0)
#             q_value_tensor = tf.expand_dims(q_value, axis=0)

#             # print("updated_q shape:", updated_q.shape)
#             # print("q_value shape:", q_value.shape)
#             loss = loss_function(updated_q_tensor, q_value_tensor)
            
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#         state = new_state
#     # if episode  == 200:
#     #     averageLength, _ = evaluateScore(model, boardDim, 25)
#     #     print(f"Episode {episode}, Average snake length without exploration: {averageLength}")

        
# #%%


# # Animate games at different episodes
# print("Generating data for animation...")
# plotEpisodes = [300, 600, 900, 1200, 1500, 1800, 2300, 2700, 3000]
# fig, axes = plt.subplots(3, 3, figsize=(9, 9))

# axList = []
# ims = []
# dataArrays = []
# scores = []
# labels = []

# for i, row in enumerate(axes):
#     for j, ax in enumerate(row):
#         ax.set_title("Episode " + str(plotEpisodes[i*len(row) + j]))
#         ax.get_yaxis().set_visible(False)
#         ax.get_xaxis().set_visible(False)
#         axList.append(ax)
#         ims.append(ax.imshow(np.zeros([boardDim, boardDim]), vmin=-1, vmax=1, cmap='RdGy'))
#         labels.append(ax.text(0, 15, "Length: 0", bbox={'facecolor': 'w', 'alpha': 0.75, 'pad': 1, 'edgecolor': 'white'}))
#         dataArrays.append(list())
#         scores.append(list())

# for episodeIndex, episode in enumerate(plotEpisodes):
#     game = SnakeGame(boardDim, boardDim)  # Assuming SnakeGame is correctly defined elsewhere
#     state = game.calcStateNum()  # Assuming this returns an integer representing the current state
#     gameOver = False

#     while not gameOver:
#         if episode <= episodeIndex:  # Random action selection for exploration
#             action = random.randint(0, 3)
#         else:
#             # Correct preparation of state_input
#             state_vector = tf.one_hot(state, depth=256)  # One-hot encode state
#             state_input = tf.expand_dims(state_vector, 0)  # Add a batch dimension

#             action_probs = model(state_input, training=False)  # Get action probabilities from the model
#             action = np.argmax(action_probs[0].numpy())  # Choose action with highest probability

#         new_state, reward, gameOver, score = game.makeMove(action)  # Assuming makeMove is correctly defined elsewhere
#         dataArrays[episodeIndex].append(game.plottableBoard())  # Assuming plottableBoard is correctly defined
#         scores[episodeIndex].append(score)
        
#         state = new_state

#     print(f"Game for episode {episode} finished, total moves: {len(dataArrays[episodeIndex])}")

# def animate(frameNum):
#     for i, im in enumerate(ims):
#         if frameNum < len(scores[i]):
#             labels[i].set_text("Length: " + str(scores[i][frameNum]))
#             ims[i].set_data(dataArrays[i][frameNum])
#     return ims + labels

# print("Animating snakes at different episodes...")

# # Find the minimum number of frames across all episodes
# numFrames = min(len(dataArray) for dataArray in dataArrays)
# ani = animation.FuncAnimation(fig, func=animate, frames=numFrames, blit=True, interval=75, repeat=False)
# plt.show(block=True)

# print("Saving to file")
# ani.save('AnimatedGames.mp4', fps=15, writer='ffmpeg', extra_args=['-vcodec', 'libx264'])
# print("Done")