



##This Explains the method of training the dataset in the training model
for epoch in range(N_epochs):
    for each batch in training_data:
        1️⃣ Forward pass  →  predict output (ŷ)
        2️⃣ Compute loss  →  compare ŷ with true labels (y)
        3️⃣ Backward pass →  compute gradients (∂loss/∂weights)
        4️⃣ Update weights → optimizer.step()
