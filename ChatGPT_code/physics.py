import tensorflow as tf

# Create TensorFlow matrices (tensors)
# Assuming you have two matrices A and B
A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
B = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

# Matrix addition
C = tf.add(A, B)  # C = A + B

# Matrix subtraction
D = tf.subtract(A, B)  # D = A - B

# Matrix multiplication
E = tf.matmul(A, B)  # E = A * B

# Element-wise multiplication
F = tf.multiply(A, B)  # F = A * B (element-wise)

# Transpose of a matrix
A_transpose = tf.transpose(A)

# Inverse of a matrix (requires tf.linalg.inv function)
A_inv = tf.linalg.inv(A)

# Determinant of a matrix (requires tf.linalg.det function)
A_det = tf.linalg.det(A)

# Identity matrix
identity_matrix = tf.eye(3, dtype=tf.float32)  # 3x3 identity matrix

# Reshape a matrix
A_reshaped = tf.reshape(A, shape=(1, 4))  # Reshape A to a 1x4 matrix

# Reduce mean along axes
mean_along_rows = tf.reduce_mean(A, axis=0)  # Calculates mean along rows
mean_along_columns = tf.reduce_mean(A, axis=1)  # Calculates mean along columns

# Print the results
print("A:\n", A.numpy())
print("B:\n", B.numpy())
print("A + B:\n", C.numpy())
print("A - B:\n", D.numpy())
print("A * B (Matrix Multiplication):\n", E.numpy())
print("A * B (Element-wise Multiplication):\n", F.numpy())
print("Transpose of A:\n", A_transpose.numpy())
print("Inverse of A:\n", A_inv.numpy())
print("Determinant of A:\n", A_det.numpy())
print("Identity Matrix:\n", identity_matrix.numpy())
print("Reshaped A:\n", A_reshaped.numpy())
print("Mean along rows:\n", mean_along_rows.numpy())
print("Mean along columns:\n", mean_along_columns.numpy())
