import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# Define the number of nodes in the input layer (assuming you flatten the matrix)
num_input_nodes = 196  # For a 14x14 matrix         

# Create a function to build the neural network
def build_neural_network():
    print("we building")
    # Input layer
    input_layer = Input(shape=(num_input_nodes,), name='input_layer')
    print("we inputted")
    # Hidden layers (you can adjust the number of layers and nodes as needed)
    hidden_layer_1 = Dense(256, activation='relu', name='hidden_layer_1')(input_layer)
    print("through layer 1")
    hidden_layer_2 = Dense(128, activation='relu', name='hidden_layer_2')(hidden_layer_1)
    print("through layer 2")
    # Output layers for node connections and original matrix
    output_connections = Dense(num_input_nodes, activation='sigmoid', name='output_connections')(hidden_layer_2)
    output_original_matrix = Dense(num_input_nodes, activation='sigmoid', name='output_original_matrix')(hidden_layer_2)
    print("output whatever thing")
    # Create the model
    model = Model(inputs=input_layer, outputs=[output_connections, output_original_matrix], name='bridge_design_model')
    print("model created")
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("model compiled")
    return model

# Build the neural network
model = build_neural_network()

# Display the model summary
model.summary()
