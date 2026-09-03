def build_classifier(n_qubits: int, layers: int = 2, shots: int | None = None):
    import pennylane as qml
    device = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(device)
    def circuit(features, weights):
        qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit, (layers, n_qubits, 3)

