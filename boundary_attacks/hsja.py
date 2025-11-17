import time
import torch
import numpy as np


def random_tangent_direction(gradient):
        """Return a random unit vector orthogonal to the gradient."""
        v = np.random.randn(*gradient.shape)
        v -= v.dot(gradient) * gradient        # remove projection onto gradient
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            return random_tangent_direction(gradient)  # retry
        return v / norm


class HSJAWithTracking:
    """
    HSJA personalizzato con tracking completo delle query
    Implementazione semplificata del paper originale
    """

    def __init__(self, model, X, scaler, max_queries=500, verbose=True):
        self.model = model
        self.scaler = scaler
        self.max_queries = max_queries
        self.verbose = verbose
        self.query_count = 0
        self.boundary_points = []
        self.all_queries = []
        self.X = X

    def predict(self, x):
        """Query con tracking"""
        self.query_count += 1
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        x_scaled = torch.from_numpy(x_scaled).float()
        pred, hidden = self.model(x_scaled)
        pred = torch.argmax(pred, dim=1)
        
        self.all_queries.append({
            'point': x.copy(),
            'hidden': hidden,
            'prediction': pred,
            'query_num': self.query_count
        })

        return pred

    def predict_with_tracking(self, x):
        """Query con tracking"""
        self.query_count += 1
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        x_scaled = torch.from_numpy(x_scaled).float()
        pred, hidden = self.model(x_scaled)
        pred = torch.argmax(pred, dim=1)
        
        self.all_queries.append({
            'point': x.copy(),
            'hidden': hidden,
            'prediction': pred,
            'query_num': self.query_count
        })

        return pred, hidden

    def binary_search(self, x0, x1, epsilon=0.01, max_iter=20):
        """Binary search"""
        label0 = self.predict(x0)

        for _ in range(max_iter):
            if np.linalg.norm(x1 - x0) < epsilon:
                break
            x_mid = (x0 + x1) / 2
            label_mid = self.predict(x_mid)

            if label_mid == label0: x0 = x_mid
            else: x1 = x_mid

        return (x0 + x1) / 2


    def estimate_gradient(self, x_boundary, n_samples=30):
        """Stima gradiente/normale alla frontiera"""
        label_boundary = self.predict(x_boundary)

        # Direzioni casuali
        directions = np.random.randn(n_samples, self.X.shape[1])
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

        epsilon = 0.05
        scores = []

        for direction in directions:
            crossings = 0
            for t in [-epsilon, -epsilon/2, 0, epsilon/2, epsilon]:
                x_test = x_boundary + t * direction

                if self.predict(x_test) != label_boundary:
                    crossings += 1

            scores.append(crossings)

        # Media pesata
        weights = np.array(scores) / (np.sum(scores) + 1e-6)
        gradient = np.sum(weights[:, None] * directions, axis=0)
        gradient = gradient / (np.linalg.norm(gradient) + 1e-9)

        return gradient


    def find_initial_adversarial(self, x_start, max_attempts=50):
        """Trova adversarial iniziale"""
        label_start, hidden = self.predict_with_tracking(x_start)

        for _ in range(max_attempts):
            if self.query_count >= self.max_queries * 0.5:
                return None

            x_random = np.random.uniform(self.X.min(0), self.X.max(0))

            output, hidden = self.predict_with_tracking(x_random)
            if output != label_start:
                return x_random, hidden

        return None, None


    def attack(self, x_start, n_iterations=100):
        """Attacco HSJA completo"""
        start_time = time.time()

        if self.verbose:
            print(f"\n[Inizializzazione] Ricerca adversarial iniziale...")
            _, manifold = self.model(torch.from_numpy(self.scaler.transform(x_start.reshape(1, -1))).float())
            print(f"   Start (manifold)={manifold.squeeze()[0]:.1f}, V={manifold.squeeze()[1]:.2f}")

        # Trova adversarial
        x_adv, manifold= self.find_initial_adversarial(x_start)

        if x_adv is None:
            print("   ✗ Adversarial iniziale non trovato")
            return np.array([])

        if self.verbose:
            print(f"   ✓ Adversarial: T={manifold.squeeze()[0]:.1f}°C, V={manifold.squeeze()[1]:.2f}Hz")

        # Proietta su boundary
        x_boundary = self.binary_search(x_start, x_adv)
        self.boundary_points.append(x_boundary)

        if self.verbose:
            print(f"\n[Iterazioni HSJA] Esplorazione lungo frontiera...")

        x_current = x_boundary

        for iteration in range(n_iterations):
            # stop condition
            if self.query_count >= self.max_queries: break
            # 1. Estimate gradient normal to the boundary
            gradient = self.estimate_gradient(x_current)
            # 2. Generate a tangent direction for ANY dimension
            tangent = random_tangent_direction(gradient)
            # 3. Alternating direction strategy 
            direction = tangent if iteration % 2 == 0 else -tangent
            # 4. Adaptive step
            step_size = 1.5 * (0.92 ** (iteration // 5))
            x_next = x_current + step_size * direction
            # 5. Bounds handling
            x_next = np.clip(x_next, self.X.min(0), self.X.max(0))
            label_current = self.predict(x_current)
            label_next = self.predict(x_next)
            # 6. If class changes, project back to boundary
            if label_next != label_current:
                # Boundary crossing
                x_boundary_new = self.binary_search(x_current, x_next)
                # avoid duplicates
                if not any(np.linalg.norm(x_boundary_new - p) < 1e-3
                        for p in self.boundary_points):
                    self.boundary_points.append(x_boundary_new)
                x_current = x_boundary_new
            else:
                x_current = x_next

            if self.verbose and len(self.boundary_points) % 5 == 0:
                        print(f"   Iter {iteration+1}: {len(self.boundary_points)} punti")

            # 7. Periodic reprojection toward boundary
            if iteration % 10 == 0 and iteration > 0:
                for alpha in [0.2, 0.5, 1.0]:
                    x_probe = x_current + alpha * gradient
                    x_probe = np.clip(x_probe, self.X.min(0), self.X.max(0))

                    if self.predict(x_probe) != label_current:
                        x_current = self.binary_search(x_current, x_probe)
                        break

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"ATTACCO COMPLETATO")
            print(f"{'='*70}")
            print(f"Query: {self.query_count}/{self.max_queries}")
            print(f"Boundary points: {len(self.boundary_points)}")
            print(f"Tempo: {elapsed:.2f}s")

        return np.array(self.boundary_points)

