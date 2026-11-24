\page foundations Foundations

MDL is a tool for deep learning, and many of its design decisions only make sense with the underlying ideas in mind.
These pages explain those ideas from the ground up — what a tensor is, how a network computes its gradients, what
the common layers do and why — and then point to where (and how) MDL implements them. They are meant to be readable
without knowing MDL, as a compact introduction to the background of deep learning; for a thorough treatment we
recommend the textbook by Goodfellow et al. \cite goodfellow2016deep.

<div class="cards">

- \subpage foundations_tensors
  Tensors, shapes, and broadcasting, and why MDL makes shapes part of the type.
- \subpage foundations_autodiff
  How gradients are computed: forward and reverse mode, and vector-Jacobian products.

</div>

\todo Pages on layers (linear maps, normalization, attention), losses and optimizers, training loops, and masking.
