import numpy as np


class RiemannianGradientAveraging:
    """
    Riemannian Gradient Descent Algorithm based on gradient averaging
    
    This class implements the Riemannian Gradient Descent Algorithm based on gradient averaging method, where the three
    geometric operations are delegated to the manifold object.
    
    The manifold object must implement:
        - project_tangent(x, v): Project vector v onto tangent space at x
        - exp(x, v): Exponential map (or retraction) at x in direction v
        - parallel_transport(u_from, x_from, x_to): Parallel transport u_from from x_from to x_to
    """
    
    def __init__(self, manifold, cost, lr, max_iter, tol):
        """
        Initialize the RiemannianGradientAveraging optimizer.
        
        Args:
            manifold: Manifold object with required methods:
            cost: Cost function f(w, X, y)
            lr: Learning rate (default: 0.1)
            max_iter: Maximum iterations (default: 50)
            tol: Tolerance for convergence (default: 1e-6)
        """
        self.manifold = manifold
        self.cost = cost
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X, y, grad_func=None, w0=None):
        """
        Fit the logistic regression model using implicit Riemannian gradient method.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            grad_func: Gradient function grad(w, X, y). If None, uses numerical gradient.
            w0: Initial weight vector. If None, randomly initialized.
        
        Returns:
            w: Optimized weight vector
            loss_hist: History of loss values
            acc_hist: History of accuracy values
        """
        # Initialize weight vector - let manifold handle initialization if needed
        if w0 is None:
            w = np.random.randn(X.shape[1])
            # Normalize only if manifold requires it (e.g., sphere)
            # For general manifolds, the exp map should handle constraints
            if hasattr(self.manifold, 'normalize'):
                w = self.manifold.normalize(w)
        else:
            w = w0.copy()
            if hasattr(self.manifold, 'normalize'):
                w = self.manifold.normalize(w)
        
        loss_hist = []
        acc_hist = []
        
        for _ in range(self.max_iter):
            # Step 1: Compute gradient at w and project to tangent space
            if grad_func is None:
                # Use numerical gradient if not provided
                grad_w = self._numerical_gradient(w, X, y)
            else:
                grad_w = grad_func(w, X, y)
            
            gk = self.manifold.project_tangent(w, grad_w)
            
            # Step 2: First exponential map step
            z = self.manifold.exp(w, -self.lr * gk)
            
            # Step 3: Compute gradient at z and project to tangent space
            if grad_func is None:
                grad_z = self._numerical_gradient(z, X, y)
            else:
                grad_z = grad_func(z, X, y)
            
            gz = self.manifold.project_tangent(z, grad_z)
            
            # Step 4: Parallel transport gradient from z to w
            gz_at_w = self.manifold.parallel_transport(gz, z, w)
            
            # Step 5: Averaged update using exponential map
            direction = -0.5 * self.lr * (gk + gz_at_w)
            w = self.manifold.exp(w, direction)
            
            # Record history
            loss_hist.append(self.cost(w, X, y))
            preds = np.sign(X @ w)
            acc_hist.append(np.mean(preds == y))
        
        return w, loss_hist, acc_hist
    
    def _numerical_gradient(self, w, X, y, eps=1e-7):
        """Compute numerical gradient (fallback if grad_func not provided)."""
        grad = np.zeros_like(w)
        for i in range(len(w)):
            w_plus = w.copy()
            w_plus[i] += eps
            w_plus /= np.linalg.norm(w_plus)
            
            w_minus = w.copy()
            w_minus[i] -= eps
            w_minus /= np.linalg.norm(w_minus)
            
            grad[i] = (self.cost(w_plus, X, y) - self.cost(w_minus, X, y)) / (2 * eps)
        return grad


class SphereManifold:
    """
    Example manifold implementation for the unit sphere.
    This demonstrates the required interface for the manifold object.
    """
    
    def project_tangent(self, x, v):
        """Project vector v onto the tangent space at x on the sphere."""
        return v - np.dot(x, v) * x
    
    def exp(self, x, v):
        """Exponential map on the sphere (generic name for any manifold)."""
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            return x
        return np.cos(norm_v) * x + np.sin(norm_v) * (v / norm_v)
    
    def parallel_transport(self, u_from, x_from, x_to):
        """Parallel transport u_from from x_from to x_to on the sphere."""
        dot_xy = np.dot(x_from, x_to)
        if 1 + dot_xy < 1e-12:  # near antipodal
            return self.project_tangent(x_to, u_from)
        coeff = np.dot(x_to, u_from) / (1 + dot_xy)
        transported = u_from - coeff * (x_from + x_to)
        return self.project_tangent(x_to, transported)
    
    def normalize(self, x):
        """Normalize vector to unit sphere (optional helper method)."""
        return x / np.linalg.norm(x)


class PymanoptManifoldAdapter:
    """
    Adapter class to use pymanopt manifolds with ImplicitLogisticRegression.
    
    This adapter wraps pymanopt manifolds (which use different method names)
    to work with our interface. You can use this to work with any pymanopt
    manifold (Sphere, Stiefel, Grassmannian, etc.).
    
    Example:
        from pymanopt.manifolds import Sphere
        pymanopt_sphere = Sphere(n)
        manifold = PymanoptManifoldAdapter(pymanopt_sphere)
        optimizer = ImplicitLogisticRegression(manifold, cost, lr, max_iter, tol)
    """
    
    def __init__(self, pymanopt_manifold):
        """
        Initialize the adapter with a pymanopt manifold.
        
        Args:
            pymanopt_manifold: A pymanopt manifold object
        """
        self.manifold = pymanopt_manifold
    
    def project_tangent(self, x, v):
        """Project vector v onto the tangent space at x."""
        return self.manifold.to_tangent_space(x, v)
    
    def exp(self, x, v):
        """Exponential map (retraction) at x in direction v."""
        return self.manifold.retraction(x, v)
    
    def parallel_transport(self, u_from, x_from, x_to):
        """Parallel transport u_from from x_from to x_to."""
        # pymanopt uses transport(x_from, x_to, u_from) - note the different order
        return self.manifold.transport(x_from, x_to, u_from)
    
    def normalize(self, x):
        """Normalize if the manifold has this method."""
        if hasattr(self.manifold, 'normalize'):
            return self.manifold.normalize(x)
        return x
