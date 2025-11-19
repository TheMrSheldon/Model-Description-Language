# Model Description Language


## Compile
```bash
mdlc model.mdl -o model.o --backend cuda
```

## Intermediate Representation and Linking
Compile to compute graph
```bash
mdlc model.mdl -o model.cg
```

Linking:
```bash
mdlc model2.mdl -o model2.o -l model.cg --backend cuda
```

## Add gradient path to model
```bash
mdlc add-gradients input.cg -o output.cg --parameters *
```

## Inspect compiled model
```bash
mdlc dump model.cg
```
Output:
```
Number of parameters: XXX
Inputs: XXX
Outputs: XXX
Memory Usage: XXX
```