\page mylib_layers_source mylib/layers.dlm

The source code of \ref mylib_layers.

\htmlonly[block]<div class="mdl"><div class="fragment"><div class="line" id="l1"><span class="lineno">1</span>/**</div>
<div class="line" id="l2"><span class="lineno">2</span> * @file</div>
<div class="line" id="l3"><span class="lineno">3</span> * @brief Layers of the test package (see tests/CMakeLists.txt, mdl_doc_golden).</div>
<div class="line" id="l4"><span class="lineno">4</span> */</div>
<div class="line" id="l5"><span class="lineno">5</span>/**</div>
<div class="line" id="l6"><span class="lineno">6</span> * @brief A scaled identity, as in \cite layernorm.</div>
<div class="line" id="l7"><span class="lineno">7</span> * @param dim the size of the last axis</div>
<div class="line" id="l8"><span class="lineno">8</span> */</div>
<div class="line" id="l9"><span class="lineno">9</span>public module <a class="code" href="mylib_layers.html#mylib_layers_Scale">Scale</a>[dim: uint] {</div>
<div class="line" id="l10"><span class="lineno">10</span>    /** @brief The factors. */</div>
<div class="line" id="l11"><span class="lineno">11</span>    public <a class="code" href="mylib_layers_source.html#l11">w</a>: float[dim]</div>
<div class="line" id="l12"><span class="lineno">12</span>    <a class="code" href="mylib_layers_source.html#l12">hidden</a>: float[dim]</div>
<div class="line" id="l13"><span class="lineno">13</span></div>
<div class="line" id="l14"><span class="lineno">14</span>    /** @brief Scales @p x. @return \f$x \odot w\f$ */</div>
<div class="line" id="l15"><span class="lineno">15</span>    public float[..., dim] <a class="code" href="mylib_layers.html#mylib_layers_Scale_forward">forward</a>(x: float[..., dim]) =&gt; x * <a class="code" href="mylib_layers_source.html#l11">w</a>;</div>
<div class="line" id="l16"><span class="lineno">16</span>}</div>
<div class="line" id="l17"><span class="lineno">17</span></div>
<div class="line" id="l18"><span class="lineno">18</span>/** @brief Doubles @p x. */</div>
<div class="line" id="l19"><span class="lineno">19</span>public float[...] <a class="code" href="mylib_layers.html#mylib_layers_twice">twice</a>(x: float[...]) =&gt; <a class="code" href="mylib__impl_source.html#l2">scale</a>(x, 2.0);</div>
<div class="line" id="l20"><span class="lineno">20</span>/** @brief Doubles a masked @p x (an overload: its own anchor). */</div>
<div class="line" id="l21"><span class="lineno">21</span>public masked float[...] <a class="code" href="mylib_layers.html#mylib_layers_twice_2">twice</a>(x: masked float[...]) =&gt; <a class="code" href="mylib__impl_source.html#l2">scale</a>(x, 2.0);</div>
<div class="line" id="l22"><span class="lineno">22</span></div>
<div class="line" id="l23"><span class="lineno">23</span>float[...] <a class="code" href="mylib_layers_source.html#l23">internal</a>(x: float[...]) =&gt; x;</div>
<div class="line" id="l24"><span class="lineno">24</span></div>
<div class="line" id="l25"><span class="lineno">25</span>/** Re-exported from the implementation. */</div>
<div class="line" id="l26"><span class="lineno">26</span>public from ._impl import <a class="code" href="mylib__impl_source.html#l2">scale</a>;</div>
</div></div>\endhtmlonly
