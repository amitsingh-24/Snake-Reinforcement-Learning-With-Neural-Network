

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

