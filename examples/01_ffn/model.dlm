
module Linear[indim: uint, outdim: uint, bias: bool] {
    W: float[8,3]/*[outdim, indim]*/
    b: float[3]/*[outdim]*/

    float[..., 3] forward(x: float[1,8] /*float[..., indim]*/) =>
        x @ W + b
        /*bias?
            no => x @ W.T
            yes => x @ W.T + b*/;
}
