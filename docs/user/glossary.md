\page glossary Glossary

The terms that the documentation uses, in alphabetical order.

<dl>
<dt>\anchor glossary_axis Axis</dt>
<dd>A dimension of a tensor; a matrix has two axes. See \ref foundations_tensors.</dd>

<dt>\anchor glossary_build Build</dt>
<dd>A set of compile options (e.g. which gradients to compile), possibly one of several named builds of the same model
in a configuration file. See \ref build_config.</dd>

<dt>\anchor glossary_ellipsis Ellipsis (<tt>...</tt>), ellipsis rank</dt>
<dd>In a type, <tt>...</tt> stands for any number of leading axes, e.g. a batch and a sequence axis. The language
places no limit on it; only a compiled function needs a fixed number, the <em>ellipsis rank</em>
(<tt>--ellipsis-rank</tt>). See \ref spec_language.</dd>

<dt>\anchor glossary_einops Einops expression</dt>
<dd>A rearrangement of the axes of a tensor, written as a pattern, e.g. <tt>x[... (h d) -&gt; h ... d | h=heads]</tt>
\cite rogozhnikov2022einops. See \ref spec_einops.</dd>

<dt>\anchor glossary_elaboration Elaboration</dt>
<dd>The evaluation of a program at compile time: instantiating modules with their hyperparameters, unrolling loops, and
choosing branches. Its result is the computation graph. See \ref dev_pipeline.</dd>

<dt>\anchor glossary_entry_point Entry point (exported function)</dt>
<dd>A <tt>public</tt> member function of a compiled module; it becomes a function of the compiled model. See
\ref spec_language.</dd>

<dt>\anchor glossary_hyperparameter Hyperparameter</dt>
<dd>A compile-time argument of a module (in <tt>[...]</tt>), e.g. the sizes of a layer. Its values come from the
configuration file. See \ref build_config.</dd>

<dt>\anchor glossary_kernel_package Kernel package</dt>
<dd>A directory with optimized implementations of operations (e.g. oneDNN) and a description of the patterns they
replace. See \ref spec_kernel_packages.</dd>

<dt>\anchor glossary_masked Masked tensor</dt>
<dd>A dense tensor together with a mask of its valid elements, e.g. a batch of sequences of different lengths padded to
the same length. Reductions ignore the invalid elements. See \ref spec_language.</dd>

<dt>\anchor glossary_module Module</dt>
<dd>A part of a model with hyperparameters, weights, and functions, e.g. <tt>Linear</tt>. See \ref spec_language and
\ref std_nn.</dd>

<dt>\anchor glossary_objective Objective</dt>
<dd>A public function that returns a scalar (a loss); compiled with <tt>--grad</tt>, it returns the loss and its
gradients in one call. See \ref example_gradients.</dd>

<dt>\anchor glossary_prelude Prelude</dt>
<dd>The part of the standard library that every file imports implicitly (elementwise functions, reductions,
activations). See \ref std_prelude.</dd>

<dt>\anchor glossary_residuals Residuals</dt>
<dd>The intermediate values of the forward pass that the backward pass needs, kept in a context buffer between the two
calls. See \ref example_gradients.</dd>

<dt>\anchor glossary_weight Weight</dt>
<dd>A trainable tensor of a module (a member with a tensor type). The compiled model does not own its weights: the
caller passes them in.</dd>
</dl>
