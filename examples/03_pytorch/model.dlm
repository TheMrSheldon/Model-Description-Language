
module Linear[] {
    W: float[3, 8]
    b: float[3]

    forward(x: float[8]) =>
        W@x + b;
}
