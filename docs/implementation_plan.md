# Phase 1 Data Preparation and Filtering
You have 15 gigabytes of data representing 6200 matches. Since not all matches are high quality, you need to filter them. You should parse the CSV files in chunks using tools like Pandas to manage memory. Remove matches that are too short. Filter out games where the winner took too much damage or missed too many attacks. Keep only the matches where the player demonstrated high skill and efficient movement.

# Phase 2 Feature Engineering
You must align the CSV columns to the model inputs and outputs.
Inputs: The state of the game including health, distances, relative coordinates, velocity, and looking direction.
Outputs: The actions the bot needs to take including forward input, left input, right input, jump, sprint, left mouse button, and mouse movement.

# Phase 3 Model Creation
You should build a deep learning model using Apple MLX. Since Minecraft combat relies heavily on reaction and timing, a sequence model like a Transformer is best. The model will take a sequence of recent frames and predict the next actions from the latest frame in the window.

# Phase 4 Training
Load the Phase 2 sequence windows from `data/processed/phase2_feature_tensors_per_file/`. Split windows by `match_id` so frames from the same match do not leak between training and validation. Shuffle matches, assign 80 percent to training and 20 percent to validation, and train the MLX sequence model with batch size 64. Use binary cross entropy for the boolean action head, cross entropy for the inventory slot head, and mean squared error for the continuous mouse-movement head. Save the best checkpoint only when validation loss improves.

# Phase 5 REST API Development
Create a high performance Python web server using FastAPI. This API will load the trained model into memory. It must accept a JSON payload containing the current game state, process it through the neural network, and return the predicted keyboard and mouse inputs in less than 50 milliseconds to maintain server tick rates.

# Phase 6 Minecraft Plugin Integration
The server plugin must collect the exact same metrics that are in your CSV file 20 times per second. It will send an asynchronous HTTP request to your REST API. Once it receives the response, it will apply the predicted movement, rotation, and clicking to the Citizens bot.

# Phase 7 Testing and Bug Fixing
First, test the bot against a stationary dummy to verify aim and basic movement. Second, test latency to ensure the API communication does not lag the Minecraft server. Third, fight the bot yourself to identify weak points like getting stuck in corners or failing to sprint reset.

# Phase 8 Iteration
Record the state data when the bot behaves poorly. Analyze these specific scenarios, adjust your training data filtering, and retrain the model to fix the bugs over time.