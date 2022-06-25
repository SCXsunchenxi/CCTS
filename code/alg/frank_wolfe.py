import numpy as np
from scipy import optimize
from scipy import linalg

def build_func_grad(jac, fun, args, eps):
    if not callable(jac):
        if jac == "2-point":
            jac = None
            print("using approx gradient")
        else:
            raise NotImplementedError("jac has unexpected value.")

    if jac is None:

        def func_and_grad(x):
            f = fun(x, *args)
            g = optimize.approx_fprime(x, fun, eps, *args)
            return f, g
    else:

        def func_and_grad(x):
            f = fun(x, *args)
            g = jac(x, *args)
            return f, g
    return func_and_grad

def minimize_frank_wolfe(
        fun,
        x0,
        lmo,
        jac="2-point",
        step="sublinear",
        args=(),
        max_iter=400,
        tol=1e-12,
        callback=None,
        verbose=0,
        eps=1e-8,
    ):
    r"""Frank-Wolfe algorithm.
    Implements the Frank-Wolfe algorithm, see , see :ref:`frank_wolfe` for
    a more detailed description.
    Args:
        fun : callable
            The objective function to be minimized.
                ``fun(x, *args) -> float``
            where x is an 1-D array with shape (n,) and `args`
            is a tuple of the fixed parameters needed to completely
            specify the function.
        x0: array-like
        Initial guess for solution.
        lmo: callable
        Takes as input a vector u of same size as x0 and returns both the update
        direction and the maximum admissible step-size.
        jac : {callable,  '2-point', bool}, optional
            Method for computing the gradient vector. If it is a callable,
            it should be a function that returns the gradient vector:
                ``jac(x, *args) -> array_like, shape (n,)``
            where x is an array with shape (n,) and `args` is a tuple with
            the fixed parameters. Alternatively, the '2-point' select a finite
            difference scheme for numerical estimation of the gradient.
            If `jac` is a Boolean and is True, `fun` is assumed to return the
            gradient along with the objective function. If False, the gradient
            will be estimated using '2-point' finite difference estimation.
        step: str or callable, optional
        Step-size strategy to use. 
            - "sublinear", will use a decreasing step-size of the form 2/(k+2). [J2013]_
            - callable, if step is a callable function, it will use the step-size returned by step(locals).
        lipschitz: None or float, optional
        Estimate for the Lipschitz constant of the gradient. Required when step="DR".
        max_iter: integer, optional
        Maximum number of iterations.
        tol: float, optional
        Tolerance of the stopping criterion. The algorithm will stop whenever
        the Frank-Wolfe gap is below tol or the maximum number of iterations
        is exceeded.
        callback: callable, optional
        Callback to execute at each iteration. If the callable returns False
        then the algorithm with immediately return.
        eps: float or ndarray
            If jac is approximated, use this value for the step size.
        verbose: int, optional
        Verbosity level.
    Returns:
        scipy.optimize.OptimizeResult
        The optimization result represented as a
        ``scipy.optimize.OptimizeResult`` object. Important attributes are:
        ``x`` the solution array, ``success`` a Boolean flag indicating if
        the optimizer exited successfully and ``message`` which describes
        the cause of the termination. See `scipy.optimize.OptimizeResult`
        for a description of other attributes.
    References:
        .. [J2013] Jaggi, Martin. `"Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization." <http://proceedings.mlr.press/v28/jaggi13-supp.pdf>`_ ICML 2013.
        .. [P2018] Pedregosa, Fabian `"Notes on the Frank-Wolfe Algorithm" <http://fa.bianp.net/blog/2018/notes-on-the-frank-wolfe-algorithm-part-i/>`_, 2018
        .. [PANJ2020] Pedregosa, Fabian, Armin Askari, Geoffrey Negiar, and Martin Jaggi. `"Step-Size Adaptivity in Projection-Free Optimization." <https://arxiv.org/pdf/1806.05123.pdf>`_ arXiv:1806.05123 (2020).
    """
    x0 = np.asanyarray(x0, dtype=np.float)
    if tol < 0:
        raise ValueError("Tol must be non-negative")
    x = x0.copy()
    step_size = None

    func_and_grad = build_func_grad(jac, fun, args, eps)

    f_t, grad = func_and_grad(x)

    it = 0
    for it in range(max_iter):
        update_direction, max_step_size = lmo(-grad, x)
        norm_update_direction = linalg.norm(update_direction) ** 2
        certificate = np.dot(update_direction, -grad)

        if certificate <= tol:
            break
        if hasattr(step, "__call__"):
            step_size = step(locals())
            f_next, grad_next = func_and_grad(x + step_size * update_direction)
        elif step == "sublinear":
            # .. without knowledge of the Lipschitz constant ..
            # .. we take the sublinear 2/(k+2) step-size ..
            step_size = 2.0 / (it + 2)
            f_next, grad_next = func_and_grad(x + step_size * update_direction)
        else:
            raise ValueError("Invalid option step=%s" % step)

        if callback is not None:
            if callback(locals()) is False:  # pylint: disable=g-bool-id-comparison
                break
        x += step_size * update_direction
        f_t, grad = f_next, grad_next

    if callback is not None:
        callback(locals())
    return optimize.OptimizeResult(x=x, nit=it, certificate=certificate)