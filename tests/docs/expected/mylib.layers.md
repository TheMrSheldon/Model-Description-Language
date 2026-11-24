\page mylib_layers mylib.layers

@brief Layers of the test package (see tests/CMakeLists.txt, mdl_doc_golden).

\tableofcontents

# Modules

| Name | Description |
|---|---|
| \ref mylib_layers_Scale "Scale" | A scaled identity, as in \cite layernorm. |

## Scale {#mylib_layers_Scale}

\htmlonly[block]<div class="memitem"><div class="memproto mdl">\endhtmlonly
```
module Scale[dim: uint]
```
\htmlonly[block]</div><div class="memdoc">\endhtmlonly

A scaled identity, as in \cite layernorm.
@param dim the size of the last axis


\htmlonly[block]<p class="definition">Definition at line <a href="mylib_layers_source.html#l9">9</a> of <a href="mylib_layers_source.html">mylib/layers.dlm</a>.</p>\endhtmlonly
<b>Members</b>

| Member | Type | Description |
|---|---|---|
| <tt>w</tt> | <tt>float[dim]</tt> | The factors. |

\htmlonly[block]</div></div>\endhtmlonly

### Scale.forward {#mylib_layers_Scale_forward}

\htmlonly[block]<div class="memitem"><div class="memproto mdl">\endhtmlonly
```
float[..., dim] forward(x: float[..., dim])
```
\htmlonly[block]</div><div class="memdoc">\endhtmlonly

Scales @p x. @return \f$x \odot w\f$


\htmlonly[block]<p class="definition">Definition at line <a href="mylib_layers_source.html#l15">15</a> of <a href="mylib_layers_source.html">mylib/layers.dlm</a>.</p>\endhtmlonly

\htmlonly[block]</div></div>\endhtmlonly

# Functions

| Name | Description |
|---|---|
| \ref mylib_layers_twice "twice" | Doubles @p x. |
| \ref mylib_layers_twice_2 "twice" | Doubles a masked @p x (an overload: its own anchor). |

## twice {#mylib_layers_twice}

\htmlonly[block]<div class="memitem"><div class="memproto mdl">\endhtmlonly
```
float[...] twice(x: float[...])
```
\htmlonly[block]</div><div class="memdoc">\endhtmlonly

Doubles @p x.


\htmlonly[block]<p class="definition">Definition at line <a href="mylib_layers_source.html#l19">19</a> of <a href="mylib_layers_source.html">mylib/layers.dlm</a>.</p>\endhtmlonly

\htmlonly[block]</div></div>\endhtmlonly

## twice {#mylib_layers_twice_2}

\htmlonly[block]<div class="memitem"><div class="memproto mdl">\endhtmlonly
```
masked float[...] twice(x: masked float[...])
```
\htmlonly[block]</div><div class="memdoc">\endhtmlonly

Doubles a masked @p x (an overload: its own anchor).


\htmlonly[block]<p class="definition">Definition at line <a href="mylib_layers_source.html#l21">21</a> of <a href="mylib_layers_source.html">mylib/layers.dlm</a>.</p>\endhtmlonly

\htmlonly[block]</div></div>\endhtmlonly

# Re-exports

- `scale` of \ref mylib__impl

The source code: \subpage mylib_layers_source "mylib/layers.dlm".
